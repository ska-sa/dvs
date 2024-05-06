"""
    This module provides the means to generate standard 'RTS'-style holography reports using katholog.
    It's intended to replace katholog.hologreport.
    
    Basic usage example:
        DISHPARAMS = dict(telescope="SKA", xmag=-f_eq/F_mr, focallength=F_mr, xyzoffsets=[0.0, 1.476396+8.04, 0]) # Phase reference position used to generate predicted patterns, katholog coordinates
        predicted = ResultSet("predicted", f_MHz=[11452], beacon_pol=["RCP"], clipextent=1.8)
        for f,pol in zip(predicted.f_MHz,predicted.beacon_pol):
            b, aH, aV = load_predicted(f, beacon_pol=pol, el_deg=45, clipextent=predicted.clipextent, xyzoffsets=DISHPARAMS["xyzoffsets"], xmag=DISHPARAMS["xmag"], focallength=DISHPARAMS["focallength"])
            predicted.beams.append(b); predicted.apmapsH.append(aH); predicted.apmapsV.append(aV)
        
        DISHPARAMS.update(xyzoffsets=[0.0, 1.49, -3.52]) # Phase reference position of the feed, katholog coordinates
        recs = [ResultSet(1630887121, f_MHz=[11452.2197], beacon_pol=["RCP"], clipextent=1.8, tags=["note 1"])]
        load_records(recs, "s0000", DISHPARAMS=DISHPARAMS, flag_slew=True, inspect_DT=True, timingoffset=0.1)
        generate_results(recs, predicted, beampolydegree=28, beamsmoothing="zernike", SNR_min=30, makepdfs=True, pdfprefix="ku")
    
    @author aph@ska.ac.za
"""
import numpy as np
import scipy.optimize as sop
import scipy.signal as sig
from collections import namedtuple
import itertools as iter
import copy
import time
import katholog
import logging; logging.disable(logging.DEBUG) # This is necessary for katdal
import katpoint
import katselib, katsemat
import zernike

import pylab as plt


FID2FN = {} # Overrid fid -> filename
TIME_OFFSETS = {} # Base timingoffset to load data with, e.g. where global synch failed.
fid2fn = lambda fid: FID2FN.get(fid, "http://archive-gw-1.kat.ac.za/%d/%d_sdp_l0.full.rdb"%(fid,fid))



def fitzernike(pattern, fitdBlevel, n, normalized=True, fill_value=np.nan):
    """ Fit a Zernike polynomial model to a 2D map.
        CAUTION: Unless there's large-scale background distortion it is best to fit on the entire pattern (don't use fitdBlevel).
        @param pattern: a 2D map, may be complex
        @param fitdBlevel: use only data above this level for fitting.
        @param n: the radial order up to & including which to generate polynomials for (starting from 0, so 2 is the first set including circular)
        @return: the smoothed beam, with 'fill_value' below 'fitdBlevel' """
    ddx = np.linspace(-1, 1, pattern.shape[1]+1)[:-1]
    ddy = np.linspace(-1, 1, pattern.shape[0]+1)[:-1]
    xv, yv = np.meshgrid(ddx, ddy)
    
    result = np.array(pattern, copy=True)
    mask = ~np.isfinite(pattern)
    if fitdBlevel:
        power = np.abs(pattern)**2
        power /= 1 if normalized else np.nanmax(power)
        mask |= (10*np.log10(power)<-abs(fitdBlevel))
        NV = 10**(-abs(fitdBlevel)/20.)/2.
    else:
        NV = np.nanpercentile(np.abs(pattern), 1) # A small enough number of points to not skew the fit
    result[mask] = NV # Apply masking
    
    # Only fit on the smallest rectangular range of values that are not masked out - speeds up the fit
    support = np.ix_(*[range(np.min(dim), np.max(dim)+1) for dim in np.nonzero(~mask)])
    cart = zernike.CZern(n) if np.iscomplexobj(result) else zernike.RZern(n)
    cart.make_cart_grid(xv[support], yv[support])
    coeff = cart.fit_cart_grid(result[support])[0]
    result[support] = cart.eval_grid(coeff, matrix=True)
    
    result[mask] = fill_value
    return result

def smooth_fourier(image, fitdBlevel, model, normalized=True, fill_value=np.nan, modelFTaperdB=14):
    """ Filter the 2D map in the Fourier domain, using a kernel derived from the model.
        @param fitdBlevel: ignored.
        @param model: the model image from which to derive the smoothing kernel (identical shape to image).
        @param modelFTaperdB: the edge of the Fourier transfer of 'model' is low by this power ratio [dB] from the peak amplitude, defines the boundary of the kernel (default 14)
        @return: the smoothed map, with NaNs below 'fitdBlevel'
    """
    assert (np.prod(image.shape) == np.prod(model.shape)), "Can't filter mismatching shapes 'image' %s & 'model' %s"%(image.shape, model.shape)
    
    mask = ~np.isfinite(image)
    if fitdBlevel: # The actual filtering is not affected by this parameter.
        power = np.abs(image)**2
        power /= 1 if normalized else np.nanmax(power)
        mask |= (10*np.log10(power)<fitdBlevel)
    
    def filter_error_beam(measured, model, th=100): # Fourier filtering proposed by MdV on 28/05/2021
        fft = np.fft
        fouriermeasured=fft.fftshift((fft.fft2(fft.fftshift(np.nan_to_num(measured)))))
        fouriermodel=fft.fftshift((fft.fft2(fft.fftshift(np.nan_to_num(model)))))
        threshold=th(np.abs(fouriermodel)) if callable(th) else th
        fouriermask=(np.abs(fouriermodel)>threshold)
        filteredmeasured=fft.fftshift(fft.ifft2(fft.fftshift(fouriermeasured*fouriermask)))
        return filteredmeasured
    filtered = filter_error_beam(image, model, th=lambda ffta: 0.5*np.max(ffta)*(10**-abs(modelFTaperdB/20.)))
    filtered[mask] = fill_value
    
    return filtered

def smoothbeam(beam, fitdBlevel, degree, kind="zernike", d_terms=False): # TODO: make zernike the default in katholog.BeamCube?
    if (kind.lower()=="zernike"): # Best fits when using all data (ignoring fitdBlevel) except in cases with very large-scale distortion
        beam.mGx = np.array([fitzernike(g, None, degree) for g in beam.Gx])
        beam.mGy = np.array([fitzernike(g, None, degree) for g in beam.Gy])
        if d_terms:
            beam.mDx = np.array([fitzernike(g, None, degree) for g in beam.Dx])
            beam.mDy = np.array([fitzernike(g, None, degree) for g in beam.Dy])
    elif (kind.lower()=="fourier"): # fitdBlevel is irrelevant; 'degree' is assumed to be '(modelx,modely)'
        beam.mGx = np.array([smooth_fourier(g, None, degree[0]) for g in beam.Gx])
        beam.mGy = np.array([smooth_fourier(g, None, degree[1]) for g in beam.Gy])
        if d_terms:
            beam.mDx = np.array([smooth_fourier(g, None, degree[0]) for g in beam.Dx])
            beam.mDy = np.array([smooth_fourier(g, None, degree[1]) for g in beam.Dy])
    else: # Polynomial fitting is typically very sensitive to fitdBlevel
        beam.fitpoly(fitdBlevel=fitdBlevel, degree=degree)


def load_predicted(freqMHz, beacon_pol, DISHPARAMS, el_deg=45, band="Ku", root="./models/BeamPatterns/ska", applypointing='perfeed', gridsize=512, **kwargs):
    """ Loads predicted holography datasets and projected to physical geometry as specified by 'DISHPARAMS'.
        Simulated patterns are converted from native "transmit" to "receive".
        File naming convention: either MK_GDSatcom_{freqMHz}.mat or {band}_{el_deg}_{freqMHz}.mat
        
        @param freqMHz: frequency for simulated patterns [MHz]
        @param beacon_pol: [e_H,e_V] basis vectors or None for an un-polarised beacon; e.g. [1,-1j] for RCP (also recognises 'RCP'& 'LCP').
        @param DISHPARAMS: {telescope, xyzoffsets, xmag, focallength} to project the patterns to the physical geometry.
        @param el_deg: elevation for simulated distortion, 45deg is essentially undistorted.
        @param kwargs: passed to katdal.Dataset e.g. 'clipextent'
        @return: (BeamCube, ApertureMap{H}, ApertureMap{V})
    """
    telescope, xyzoffsets, xmag, focallength = DISHPARAMS["telescope"], DISHPARAMS["xyzoffsets"], DISHPARAMS["xmag"], DISHPARAMS["focallength"]
    
    ff = freqMHz - int(freqMHz)
    try:
        try:
            dataset = katholog.Dataset("%s/MK_GDSatcom_%s_%d%d.mat"%(root,band,freqMHz,ff*10), telescope, freq_MHz=freqMHz, method='raw', **kwargs)
        except IOError:
            dataset = katholog.Dataset("%s/MK_GDSatcom_%d%d.mat"%(root,freqMHz,ff*10), telescope, freq_MHz=freqMHz, method='raw', **kwargs)
    except IOError:
        ff = "" if (ff==0) else "_%d"%(ff*10)
        dataset = katholog.Dataset("%s/%s_%d_%d%s.mat"%(root,band,el_deg,freqMHz,ff), telescope, freq_MHz=freqMHz, method='raw', **kwargs)
    # Conjugation changes the direction of travel (+z); then invert the 'll' axis to maintain IEEE definition of RCP.
    dataset.visibilities = [np.conj(v) for v in dataset.visibilities]
    dataset.ll = -dataset.ll # This is required to preserve the sign definition of circular pol (flips aperture pattern horizontally).
    
    # Don't flip any dimension in ApertureMap as that obstructs understanding - flips relate to polarisation!
    flip = dict(flipx=False,flipy=False,flipz=False)
    # NB: This "legacy flip" used to change katholog's definition of '+mm' for predicted patterns to match '+mm' for measured patterns.
    #      Now moved to 'load_data()' to rather get definition of '+mm' for measured patterns to match '+mm' for predicted patterns.
    #dataset.mm = -dataset.mm
    
    beamcube = katholog.BeamCube(dataset, xyzoffsets=xyzoffsets, applypointing=applypointing, interpmethod='scipy', gridsize=gridsize)
    if (beacon_pol is not None):
        beacon_pol = [1,-1j] if (beacon_pol == "RCP") else ([1,1j] if (beacon_pol == "LCP") else beacon_pol)
        fcH = dict(feedcombine=[beacon_pol[0],beacon_pol[1],0,0]) # feedcombine: [Gx, Dx, Dy, Gy]
        fcV = dict(feedcombine=[0,0,beacon_pol[0],beacon_pol[1]]) # feedcombine: [Gx, Dx, Dy, Gy]
        # Modify Gx & Gy to match measured, since measured patterns include the polarisation state of the beacon.
        # With this approach the only sensible applypointing seems to be 'perfeed' for everything (incl. measured - see 'load_data()')
        _H = katholog.BeamCube(dataset, xyzoffsets=xyzoffsets, applypointing=applypointing, interpmethod='scipy', gridsize=gridsize, **fcH)
        _V = katholog.BeamCube(dataset, xyzoffsets=xyzoffsets, applypointing=applypointing, interpmethod='scipy', gridsize=gridsize, **fcV)
        beamcube.Gx = _H.Gx; beamcube.Gy = _V.Gy
        beamcube.Dx = 0*_H.Dx; beamcube.Dy = 0*_V.Dy # We use feedcombine to get beam(XX) = G + D so then the D terms must be zeroed to avoid potential double accounting  
    else:
        fcH = dict(feed="H")
        fcV = dict(feed="V")

    apmapH = katholog.ApertureMap(dataset, xyzoffsets=xyzoffsets, feedoffset=None, xmag=xmag,focallength=focallength, gridsize=gridsize, **{k:v for d in (fcH,flip) for k,v in d.items()})
    apmapV = katholog.ApertureMap(dataset, xyzoffsets=xyzoffsets, feedoffset=None, xmag=xmag,focallength=focallength, gridsize=gridsize, **{k:v for d in (fcV,flip) for k,v in d.items()})
    
    return (beamcube, apmapH, apmapV)


def e_bn(pol, deg):
    """ @param pol: "H" | "V" for 100% linear | "HV" for partial linear
        @param deg: angle [deg] by which to rotate the aperture plane to align +V to NCP relative to the source.
                    When looking from the aperture to the source, +V is "up", +H is to the east and this angle is anticlockwise.
                    So if observer's latitude is south of the declination of the source then there's an extra 180deg.
        @return: [e_H, e_V] components
    """
    if (pol == "H"):
        return [np.cos(deg*np.pi/180), -np.sin(deg*np.pi/180)]
    elif (pol == "V"):
        return [np.sin(deg*np.pi/180), np.cos(deg*np.pi/180)]
    elif (pol == "HV"):
        return [np.cos(deg*np.pi/180), -np.sin(deg*np.pi/180), np.sin(deg*np.pi/180), np.cos(deg*np.pi/180)]


def load_data(fn, freqMHz, scanant, DISHPARAMS, timingoffset=0, polswap="", dMHz=0.1, load_cycles=None, overlap_cycles=0, flag_slew=False, applypointing='perfeed', gridsize=512, debug=False, **kwargs):
    """ Loads measured holography datasets for the specified telescope, projected to physical geometry as specified.
        
        @param fn: the filename or URL for the dataset.
        @param freqMHz: frequency for measured patterns [MHz], must be acurate to within +/-dMHz.
        @param DISHPARAMS: {telescope, xyzoffsets, xmag, focallength} to project the patterns to the physical geometry.
        @param timingoffset: used to adjust the time offset between signal and pointing coordinates, passed to katholog.Dataset (default 0).
        @param polswap: "0" | "1" | "0,1" if the polarisations of any of the antennas must be swapped (0=scan|track, 1=track|scan: baseline ordering is not "regular",
                       but m0xx will always be 0 if the other antenna is s0xxx) (default None)
        @param load_cycles: indices of individual cycles to load e.g. [0,1,3] or None to load it as a single cycle (default None)
        @param overlap_cycles: how many additional cycles to combine into a single map (default 0)
        @param flag_slew: True to ensure "slew" activities are always flagged out, otherwise may include "slew" if quality is OK (default False).
        @param kwargs: passed to katdal.Dataset e.g. 'timingoffset', 'clipextent'
        @return: [beams], [apmapsH], [apmapsV] to match dimensions of freqMHz, and if present, cycles.
                 Note: each beam is given the following extra attributes: {time_avg, deg_per_sec, el_deg, sun_deg, sun_rel_deg, temp_C, wind_mps, wind_rel_deg, feedindexer_deg, rawonboresight}
    """
    telescope, xyzoffsets, xmag, focallength = DISHPARAMS["telescope"], DISHPARAMS["xyzoffsets"], DISHPARAMS["xmag"], DISHPARAMS["focallength"]
    dataset = katholog.Dataset(fn, telescope, scanantname=scanant, method='gainrawabs', timingoffset=timingoffset, **kwargs)
    dataset.band = dataset.h5.spectral_windows[0].band # Cache this value for later reference
    if polswap: # Correct for polarisation swap(s) at scan antenna
        # Unfortunately the following is not usable for katdal datasets
        # dataset.visibilities = [dataset.visibilities[i] for i in [2,3,0,1]] # ['(V)H','(V)V','(H)H','(H)V'] -> ['(H)H','(H)V','(V)H','(V)V']
        # Either change dataset.pols_to_use or equivalently overload dataset.getvisslice() & swap the order of its outputs.
        
        # dataset.pols_to_use is e.g. ['HH','HV','VH','VV'], pairs are either (scan,track) OR (track,scan); the code maps this in **fixed order** to [xx, xy, yx, yy]
        # According to MdV katdal baseline ordering is not "regular" but makes m0xx always before s0xxx, regardless of which antenna scans.
        # TODO: use dataset.h5.corr_products to automate this mapping!
        if (polswap=="0"): POL_ORDER = [({'H':'V','V':'H'}[p[0]])+p[1] for p in dataset.pols_to_use] # First antenna in the pair has the swap
        if (polswap=="1"): POL_ORDER = [p[0]+({'H':'V','V':'H'}[p[1]]) for p in dataset.pols_to_use] # Second antenna in the pair has the swap
        if (polswap=="0,1"): POL_ORDER = [({'H':'V','V':'H'}[p[0]])+({'H':'V','V':'H'}[p[1]]) for p in dataset.pols_to_use] # Both antennas have the swap
        
        POL_ORDER = [list(dataset.pols_to_use).index(p) for p in POL_ORDER] # Convert to indices
        dataset.pols_to_use = [dataset.pols_to_use[p] for p in POL_ORDER] # Swap the order used by all loading & processing
        
    def _load_extrainfo_(dataset, f_MHz, dMHz, out): # Add attributes to 'out', based on dataset as currently flagged.
        out.time_avg = dataset.env_times[0]
        out.deg_per_sec = np.percentile(dataset.deg_per_min, 95, axis=0)/60. # (Az,El)
        az_deg = np.mean(dataset.h5.az)
        out.el_deg = dataset.env_el[0]
        out.parangle_deg = np.median((dataset.h5.parangle+360)%360) # +/-180
        out.temp_C = dataset.env_temp # (avg,min,max)
        out.wind_mps = dataset.env_wind # (avg,min,max)
        out.wind_rel_deg = katsemat.wrap(dataset.env_wind_dir[0]-az_deg, 360) # mean relative az. env_wind_dir[0] (mean direction) is on the same range as 'target azel', while [1,2] are over(0,360) so would need another 'wrap' to avoid 180deg ambiguity
        out.sun_deg = dataset.env_sun # (avg,min,max) distance from bore sight
        sun_azel = np.array(katpoint.Target('Sun, special').azel(out.time_avg, antenna=dataset.h5.ants[0]))*180/np.pi # (az,el) mean angle [deg]
        out.sun_rel_deg = ([katsemat.wrap(sun_azel[0]-az_deg, 360), sun_azel[1]-out.el_deg]) if (sun_azel[1] > -5) else (np.nan, np.nan) # (az,el) mean angle relative to bore sight
        
        # Feed Indexer angles
        scanant = dataset.radialscan_allantenna[dataset.scanantennas[0]]
        a = [np.nan]*3
        try: # TODO: consider changing below to katdal's equivalent mechanism
            if (telescope.upper() == "MEERKAT"):
                a = katselib.getsensorvalues("%s_ap_indexer_position_raw"%scanant, dataset.rawtime)[1]
            elif (telescope.upper() == "SKA"):
                a = katselib.getsensorvalues("%s_dsm_indexerActualPosition"%scanant, dataset.rawtime)[1]
        except: # Sensor database may not be available
            pass
        out.feedindexer_deg = np.round([np.mean(a), np.min(a), np.max(a)],6) # (avg,min,max)
        
        # 'rawonboresight'
        if True: # Indented to show this is unaffected by dataset.flagdata(), consider loading only once
            trackant = dataset.radialscan_allantenna[dataset.trackantennas[0]]
            polproducts = [("%s%s"%(scanant,p[0].lower()), "%s%s"%(trackant,p[1].lower())) for p in dataset.pols_to_use] # Potentially different order: originally [HH,HV,VH,VV]
            cpindices = []
            for p in polproducts: # The correlator's ordering matters and some times it is the other way round
                cpi = np.all(dataset.h5.corr_products==p,axis=1) | np.all(dataset.h5.corr_products==list(reversed(p)), axis=1)
                cpindices.append(np.flatnonzero(cpi)[0])
            polproducts = [p[0]+'-'+p[1] for p in polproducts] # Convert to simpler format for reporting.
            if polswap: # Swap the order of the labels if necessary
                polproducts = [polproducts[p] for p in POL_ORDER]
            bore_dumps = (dataset.ll)**2+(dataset.mm)**2 < (dataset.radialscan_sampling)**2
        dumps = bore_dumps[dataset.time_range]
        timestamps = dataset.h5.timestamps[dumps]
        chan = dataset.h5.channels[np.abs(dataset.h5.channel_freqs/1e6-f_MHz) <= dMHz]
        rawonaxis = [dataset.h5.vis[dumps,chan,p] for p in cpindices] # (prod, time, freq)
        out.rawonboresight = (timestamps, np.mean(rawonaxis, axis=2), polproducts) # [1]=(prod, time)
    
    def _load_cycle_(f_MHz, b_buf, aH_buf, aV_buf):
        # Don't flip any dimension in ApertureMap as that obstructs understanding
        flip = dict(flipx=False,flipy=False,flipz=False)
        # NB: flip the sign of 'mm' to get definition of '+mm' for measured patterns to match '+mm' for predicted patterns.
        #      katholog registers +mm (+dEl) when antenna points above target, while measuring the lower part of the pattern, which is -mm for predicted patterns.
        dataset.mm = -dataset.mm
        # Measured patterns are 'as-received patterns' i.e. include the polarisation state of the beacon. Instead of correcting
        # these, the adopted approach is to generate predicted (beam & aperture) patterns to match the measured patterns.
        # With this approach the only sensible applypointing seems to be 'perfeed' for everything (see note under 'load_predicted()').
        # TODO: polswap does not seem to affect BeamCube!
        b_buf.append(katholog.BeamCube(dataset, scanantennaname=scanant, freqMHz=f_MHz, dMHz=dMHz, applypointing=applypointing, interpmethod='scipy', xyzoffsets=xyzoffsets, gridsize=gridsize))
        aH_buf.append(katholog.ApertureMap(dataset, scanantennaname=scanant, xyzoffsets=xyzoffsets, feed='H', freqMHz=f_MHz, dMHz=dMHz, xmag=xmag,focallength=focallength, gridsize=gridsize, **flip))
        aV_buf.append(katholog.ApertureMap(dataset, scanantennaname=scanant, xyzoffsets=xyzoffsets, feed='V', freqMHz=f_MHz, dMHz=dMHz, xmag=xmag,focallength=focallength, gridsize=gridsize, **flip))
        _load_extrainfo_(dataset, f_MHz, dMHz*1.1, out=b_buf[-1]) # *1.1 to be similar to rounding applied in BeamCube & ApertureMap - without this sometimes we get the single channel adjacent to the beacon!
        dataset.mm = -dataset.mm # WIP: restore original sense of "mm" so that we don't break 'Dataset.findcycles()'
    
    clip = kwargs.get("clipextent",None)
    if (load_cycles is not None): # Identify & load ALL cycles
        icycleoffset=0; timestart_hrs=0; timeduration_hrs=1e300; maxcycles=99
        print('Finding cycles using timestart_hrs %g, timeduration_hrs %g, icycleoffset %s'%(timestart_hrs,timeduration_hrs,icycleoffset))
        # Try different combinations to find the one that best detects cycles
        cyclestart, cyclestop, nscanspercycle = [], [], 9e9
        print('Attempting to find the best combination of flagslew & onradial...')
        for _onradial,_flagslew in [([0.5, 0.75],False), ([0.25,0.5],False), ([0.05,0.2],False), ([0.05,0.2],True)]: # Defaults then progressivley more robust against poor pointing
            _flagslew = _flagslew or flag_slew # Allow override
            dataset.flagdata(timestart_hrs=timestart_hrs, timeduration_hrs=timeduration_hrs, clipextent=clip, flagslew=_flagslew)
            try:
                _cyclestart,_cyclestop,_nscanspercycle = dataset.findcycles(cycleoffset=icycleoffset, onradial=_onradial, doplot=debug)
                if (len(_cyclestart) > len(cyclestart)) or (len(_cyclestart) == len(cyclestart) and _nscanspercycle > nscanspercycle): # Best so far
                    onradial, flagslew = _onradial, _flagslew
                    cyclestart, cyclestop, nscanspercycle = _cyclestart, _cyclestop, _nscanspercycle
            except:
                pass
        if (len(cyclestart) > 0):
            overlap_cycles = int(overlap_cycles) if (overlap_cycles < len(cyclestart)) else len(cyclestart)-1
            print("Proceeding with cycles detected using flagslew %s, onradial %s. Each cycle overlaps %d subsequent cycles!"%(flagslew,onradial,overlap_cycles))
            cycles = zip(cyclestart, cyclestop[overlap_cycles:])
            beams, apmapsH, apmapsV = [[] for _ in np.atleast_1d(freqMHz)], [[] for _ in np.atleast_1d(freqMHz)], [[] for _ in np.atleast_1d(freqMHz)]
            for ic in [n for n in range(min([len(cycles), maxcycles])) if (n in load_cycles)]: # Filter as per load_cycles
                cycle = cycles[ic]
                print('--------------------------------------------\nProcessing cycle %d of %d (%.2f hrs)\n'%(ic+1,len(cycles),cycle[1]-cycle[0]))
                dataset.flagdata(timestart_hrs=cycle[0], timeduration_hrs=cycle[1]-cycle[0], clipextent=clip, flagslew=flagslew)
                for i,f_MHz in enumerate(np.atleast_1d(freqMHz)):
                    _load_cycle_(f_MHz, beams[i], apmapsH[i], apmapsV[i])
            
            dataset.h5 = None # This object is not used further and since it is not serializable, it complicates use cases 
            return beams, apmapsH, apmapsV
        else:
            print("No cycles found, fall back as if load_cycles=None")
            dataset.flagdata(clipextent=clip, flagslew=flag_slew) # Reset
    else:
        dataset.flagdata(clipextent=clip, flagslew=flag_slew) # Reset
    
    # Just load the data as currently selected.
    beams, apmapsH, apmapsV = [], [], []
    for f_MHz in np.atleast_1d(freqMHz):
        _load_cycle_(f_MHz, beams, apmapsH, apmapsV)
    
    dataset.h5 = None # This object is not used further and since it is not serializable, it complicates use cases 
    return beams, apmapsH, apmapsV


# TODO EVENTUALLY: incorporate the following modification to katholog.aperture.ApertureMap.analyse()?
def re_analyse(self, feedoffset=None, feedphasemap=0):
    """ Exactly katholog.ApertureMap.analyse() but with modifications marked as "[2]".
        
        Conceptually
                               phasemap = feedphasemap + (feedoffsetphasemap + opticsphasemap) + pointingphasemap
                      unwrappedphasemap = unwrap(phasemap)
                     nopointingphasemap = flatphase(unwrappedphasemap,'nopointing') ~ feedphasemap + smallscale_aperturephasemap + collimationphasemap
                           flatphasemap = flatphase(unwrappedphasemap,'flat') ~ feedphasemap + smallscale_aperturephasemap
        The reason for this is that flatphase(unwrappedphasemap,'flat') solves for both pointing & feed offsets simultaneously,
        but they are degenerate and so may yield artificial results that cancel to some extent.
        
        Interpretation seems to be consistent everywhere that
           (a) nopointingphasemap == opticsphasemap == reflector surfaces + collimation errors
           (b) flatphasemap == reflector surfaces
        (completely neglecting the feed contribution i.e. assumes feedphasemap = 0.)
        
        We introduce 'feedphasemap' as the total phase of the actual feed when perfectly installed on perfect optics & perfectly pointed.
        Modification [2] changes the analysis like so
                      unwrappedphasemap = unwrap(phasemap) - feedphasemap
        
        The interpretation remains unchanged, and the contribution of the feed can now truly be neglected.
        
        
        Use like 'mod_apmap = re_analyse(apmap, feedphasemap=simulated.unwrappedphasemap)'.
        This does not modify the state of 'self' ('apmap' in example usage).
        Re-analysis can be un-done by calling 'mod_apmap.analyse()'.
        
        @param feedoffset: None or a list (solver prints an error message if np.array?)
        @param feedphasemap: If not None or 0, subtract this pattern before deriving flatphasemap (default 0)
        @return: a modified (shallow) copy of 'self'
    """
    # Some trivial added code
    mod = 2 # 0=original, 2=modification developed in 2020.
    self = copy.copy(self) # Don't need deep copy, and visibilities datasets cannot be deep copied
    feedphasemap = 0 if (feedphasemap is None) else feedphasemap
    
    self.ampmap=np.abs(self.apert)
    self.phasemap=np.angle(self.apert)
    self.unwrappedphasemap=katholog.aperture.unwrap(self.ampmap,self.phasemap,self.unwrapmaskmap)
    
    if (mod == 2): # Latest modified code: remove feed phase in the unwrappedphasemap
        self.unwrappedphasemap=self.unwrappedphasemap - feedphasemap
    
    self.ampmodelmap=katholog.aperture.ampmodel(self.ampmap,self.blockdiameter,self.dishdiameter,self.mapsize,self.gridsize)
    self.nopointingphasemap,self.nopointingphaseoffset,self.nopointingphasegradient,dud,self.nopointingphaseoffsetstd,self.nopointingphasegradientstd,dud,self.nopointingfuncs=katholog.aperture.flatphase(self.ampmap if (self.fitampmap is None) else self.fitampmap,self.unwrappedphasemap,self.flatmaskmap,self.blockdiameter,self.dishdiameter,self.mapsize,self.gridsize,self.wavelength,self.focallength,self.xmag,feedoffset,self.parabolaoffset,self.flatmode,self.copolmap,self.crosspolmap)
    self.flatphasemap,self.phaseoffset,self.phasegradient,self.feedoffset,self.phaseoffsetstd,self.phasegradientstd,self.feedoffsetstd,self.funcs=katholog.aperture.flatphase(self.ampmap if (self.fitampmap is None) else self.fitampmap,self.unwrappedphasemap,self.flatmaskmap,self.blockdiameter,self.dishdiameter,self.mapsize,self.gridsize,self.wavelength,self.focallength,self.xmag,feedoffset,self.parabolaoffset,'flat')
    self.modelmap=self.unwrappedphasemap-self.flatphasemap
    self.nopointingmodelmap=self.unwrappedphasemap-self.nopointingphasemap
    self.nopointingdevmap=self.blank(katholog.aperture.getdeviation(self.nopointingphasemap,self.mapsize,self.gridsize,self.wavelength,self.focallength,self.parabolaoffsetdev))
    self.devmap=self.blank(katholog.aperture.getdeviation(self.flatphasemap,self.mapsize,self.gridsize,self.wavelength,self.focallength,self.parabolaoffsetdev))
    self.rms0_mm=self.halfpatherrorms(self.ampmap,self.nopointingphasemap)
    self.rms_mm=self.halfpatherrorms(self.ampmap,self.flatphasemap)
    self.gain(None,1.0)
    return self


def BDF(apmap, D, f):
    """ Computes the Beam Deviation Factor [Y.T. Lo "On the BDF of a Parabolic Reflector] as quoted in [Ruze " Small Displacements of Parabolic Reflectors"].
        Employs only aperture plane amplitude, so does not depend on the focal length & magnification factor with which the aperture map
        has been initialised.
        
        @param apmap: a katholog ApertureMap
        @param D,f: characteristics of the effective parabola
        @return: beam deviation factor """
    if (apmap is None):
        BDF0 = lambda k=0.36: (1+k*(D/(4*f))**2)/(1+(D/(4*f))**2) # From Y.T.Lo, k~0.36 depends on taper, f/D for the actual parabola!
        BDF = BDF0()
    else:
        x,y = np.meshgrid(np.linspace(-apmap.mapsize/2.0,apmap.mapsize/2.0,apmap.gridsize+1)[:-1],np.linspace(-apmap.mapsize/2.0,apmap.mapsize/2.0,apmap.gridsize+1)[:-1])
        r = np.sqrt(x**2+y**2)
        mask = (r<=D/2.)
        ap_ill, r = apmap.ampmap[mask], r[mask] # Only magnitude is required in the integral
        BDF = np.sum((ap_ill*r**3)/(1+(r/2./f)**2)) / np.sum(ap_ill*r**3)
    return BDF


# Data structure for processed datasets
_ResultSet_ = namedtuple("_ResultSet_", ["fid","f_MHz","beacon_pol","beams","apmapsH","apmapsV","clipextent","cycles","overlap_cycles","polswap","tags"]) # Work-around for defaults keyword not available in current version of python
ResultSet = lambda fid,f_MHz,beacon_pol,beams=0,apmapsH=0,apmapsV=0,clipextent=None,cycles=None,overlap_cycles=0,polswap=False,tags=None: _ResultSet_(
                fid,f_MHz,beacon_pol,[] if beams==0 else beams,[] if apmapsH==0 else apmapsH,[] if apmapsV==0 else apmapsV,clipextent,cycles,overlap_cycles,polswap,[] if tags==None else tags)

def load_records(holo_recs, scanant, timingoffset, DISHPARAMS, clipextent=None, flag_slew=False, inspect_DT=False,  gridsize=512, flush=False, **load_kwargs):
    """ Loads the datasets into the records' results fields. Uses global TIME_OFFSETS.
        Results are loaded into holo_recs(ResultSet).beams, apmapsH & apmapsV and have the same shape as (ResultSet).cycles.
        
        @param holo_recs: a list of 'ResultSet's
        @param timingoffset: combined with global 'TIME_OFFSETS' & passed to 'load_data()'
        @param clipextent: override the value defined in the records' 'clipextent' attribute (default None)
        @param flag_slew: True to ensure "slew" activities are always flagged out, otherwise may include "slew" if quality is OK (default False).
        @param inspect_DT: True to load data also with timingoffset=0 to generate a debug plot (default False)
    """
    global TIME_OFFSETS
    for rec in holo_recs:
        filename = fid2fn(rec.fid)
        T0 = TIME_OFFSETS.get(rec.fid, 0)
        clip = rec.clipextent if (clipextent is None) else clipextent
        if flush:
            del rec.beams[:], rec.apmapsH[:], rec.apmapsV[:]
        if (len(rec.beams) == 0): # Don't overwrite previous results
            load_cycles = rec.cycles if ((rec.cycles is None) or not isinstance(rec.cycles, int)) else [rec.cycles]
            b, aH, aV = load_data(filename, rec.f_MHz, scanant, DISHPARAMS=DISHPARAMS, timingoffset=T0+timingoffset, polswap=rec.polswap, flag_slew=flag_slew, gridsize=gridsize, clipextent=clip, load_cycles=load_cycles, overlap_cycles=rec.overlap_cycles, **load_kwargs)
            if isinstance(rec.cycles, int):
                b, aH, aV = np.asarray(b)[:,0], np.asarray(aH)[:,0], np.asarray(aV)[:,0] # Leave the first dimension, which is frequency
            rec.beams.extend(b); rec.apmapsH.extend(aH); rec.apmapsV.extend(aV)
            if inspect_DT:
                cycle0 = 0 if (load_cycles is None) else load_cycles[0]
                b0 = load_data(filename, rec.f_MHz[0], scanant, DISHPARAMS=DISHPARAMS, timingoffset=T0, polswap=rec.polswap, flag_slew=flag_slew, gridsize=gridsize, clipextent=clip, load_cycles=[cycle0], overlap_cycles=rec.overlap_cycles, **load_kwargs)[0]
                plt.figure(figsize=(12,6))
                plt.subplot(1,2,1); np.atleast_1d(b0[0])[0].plot("Gx", doclf=False); plt.title("%d: timingoffset = 0"%rec.fid) # First frequency, first cycle
                plt.subplot(1,2,2); np.atleast_1d(b[0])[0].plot("Gx", doclf=False); plt.title("timingoffset = %g"%timingoffset)
        else:
            print(filename)
        print(rec)


# Data structure for results generated by 'standard_report()'
HologResults = namedtuple("HologResults",["el_deg","f_MHz","feedoffsetsH","feedoffsetsV","rpeffH","rpeffV",
                                          "rmsH","rmsV","errbeamH","errbeamV","info"])

def _unsqueeze_(rs):
    """ Add back the redundant dimensions to HologResults that standard_report() squeezes out for "1 cycle" sets
        @return: a copy of rs [HologResults] """
    nest1d = lambda x: np.reshape(x, -1); nest2d = lambda x: np.reshape(x, (-1,np.shape(x)[-1])); nest3d = lambda x: np.reshape(x, (1,-1,np.shape(x)[-1]))
    RS = HologResults(nest1d(rs.el_deg), rs.f_MHz, nest3d(rs.feedoffsetsH), nest3d(rs.feedoffsetsV),
                      nest3d(rs.rpeffH), nest3d(rs.rpeffV), nest2d(rs.rmsH), nest2d(rs.rmsV), nest3d(rs.errbeamH), nest3d(rs.errbeamV),
                      dict([(k,nest1d(v)) for k,v in rs.info.items()]))
    return RS

def _plan_layout_(RS, labels, separate_freqs):
    """ Transposes the data in RS against frequency, optionally structuring it separately by frequency.
        @param RS: a list of one or more HologResults
        @param labels: a list of labels for each item in RS
        @param separate_freqs: False to lump all frequencies together 
        @return: nested lists of (f_index, RS, label) or ([f0,f1,...]_index, [RS0,RS1,...], label) """
    layout = []
    for rs,lbl in zip(RS,labels):
        lbl = ("("+lbl+") ") if lbl else ""
        freqs = []
        for r in rs:
            freqs.extend(r.f_MHz)
        freqs = sorted(set(freqs))
        if separate_freqs: # Each unique frequency in a different panel
            for f in freqs:
                _rs = [r for r in rs if (f in r.f_MHz)]
                layout.append([[r.f_MHz.index(f) for r in _rs], _rs, lbl])
        else: # All frequencies in one panel
            _f, _rs = [], []
            for f in freqs:
                __rs = [r for r in rs if (f in r.f_MHz)]
                _rs.extend(__rs)
                _f.extend([r.f_MHz.index(f) for r in __rs])
            layout.append([_f,_rs,lbl])
    return layout


def plot_vs_hod(RS, labels, separate_freqs=True, fspec_MHz=(15000,20000), eff_ix=-1, EBextra="RMS", figsize=(14,4)):
    """ Generates a figure of the following metrics vs. hour of day for cycles:
        * sun distance from bore sight, wind speed
        * sun & wind attack angles relative to bore sight
        * feed offsets,
        * phase efficiency at highest freq, and
        * error beam
        
        @param RS: a list of lists of 'HologResults'
        @param labels: a text label for each set of results
        @param separate_freqs: True to have unique measurement frequencies in different panels (default True)
        @param fspec_MHz: the frequencies corresponding to the last columns in phase_eff [MHz]
        @param eff_ix: selects which efficiency results to plot (as a negative index into 'fspec_MHz') (default -1) """
    layout = _plan_layout_([[_unsqueeze_(r) for r in rs] for rs in RS], labels, separate_freqs)
    
    axes = np.atleast_1d(plt.subplots(5*len(layout),1,sharex=True,figsize=(figsize[0],figsize[1]*5*len(layout)))[1])
    for i,(fs,rs,lbl) in enumerate(layout):
        # For each HologResults in 'rs', indices for cycles that are 'masked out'. feedoffsets & errbeam share the same mask
        mask_ix = [[] if not hasattr(r.feedoffsetsH, "mask") else [ci for ci in range(len(r.el_deg)) if (np.all(r.feedoffsetsH[i,ci,...].mask) and np.all(r.feedoffsetsV[i,ci,...].mask))] for r in rs]
        
        # Set the title for the top most panel for this series
        ax = axes[5*i]
        el_deg = [np.mean(r.el_deg) for r in rs] # Mean elevation for each set in this series
        if (np.max(el_deg)-np.min(el_deg) > 1):
            ax.set_title("Results collected at %.1f-%.1fdeg elevation"%(np.min(el_deg), np.max(el_deg)))
        else:
            ax.set_title("Results collected at %.1fdeg elevation"%np.mean(el_deg))
        # Sun distance & wind speed
        ax_ = ax.twinx()
        for r,mi in zip(rs,mask_ix):
            time_hod = r.info["time_hod"]
            ax.plot(time_hod, [np.min(e["sun_deg"]) for e in r.info["enviro"]], 'r*')
            ax_.plot(time_hod, [e["wind_mps"][0] for e in r.info["enviro"]], 'b_') # avg
            if (len(mi)>0): # Overplot special symbols where all results are masked out
                ax.plot(np.take(time_hod,mi), np.take([np.min(e["sun_deg"]) for e in r.info["enviro"]],mi), 'ks', alpha=0.5)
                ax_.plot(np.take(time_hod,mi), np.take([e["wind_mps"][0] for e in r.info["enviro"]],mi), 'ks', alpha=0.5)
            if (len(time_hod) > 1):
                wrp = np.argmin(np.sign(np.diff(time_hod))) # Index position where time crossed over midnight
                wrp = [slice(0,None)] if (wrp == 0) else [slice(0,wrp+1), slice(wrp+1,None)]
                for ix in wrp:
                    ax_.fill_between(time_hod[ix], [e["wind_mps"][1] for e in r.info["enviro"][ix]], [e["wind_mps"][2] for e in r.info["enviro"][ix]], facecolor='b', alpha=0.1)
        ax.set_ylabel("Sun proximity to bore sight [deg]", color="r"); ax_.set_ylabel("Max avg wind speed [m/s]", color="b"); ax.grid(True)
        # Sun & wind angles in tangent plane directed **TOWARDS** bore sight
        ax = axes[5*i+1]
        for r,mi in zip(rs,mask_ix):
            time_hod = np.asarray(r.info["time_hod"])
            # Sun & wind angles are from boresight to sun & to wind origin, but attack vector is from those to boresight so change sign of l & m
            ae2lm = lambda Daz,Del,el: -np.asarray(katpoint._projection.sphere_to_ortho(az0=0,el0=el*np.pi/180,az=Daz*np.pi/180,el=(el+Del)*np.pi/180)[:2]) # deg,deg,deg -> -l,-m in tangent plane on bore sight
            sun_ae = np.transpose([np.asarray(e["sun_rel_deg"]) for e in r.info["enviro"]]) # dAz,dEl from bore to sun
            ax.quiver(time_hod, 0*time_hod, *ae2lm(sun_ae[0], sun_ae[1], r.el_deg), scale=None, color='r', alpha=0.5)
            wind_ae = (np.asarray([e["wind_rel_deg"] for e in r.info["enviro"]]), (0-r.el_deg)) # dAz,dEl from bore to wind origin
            ax.quiver(time_hod, 0*time_hod, *ae2lm(wind_ae[0], wind_ae[1], r.el_deg), scale=None, color='b', alpha=0.5)
            if (len(mi)>0): # Overplot special symbols where all results are masked out
                ax.quiver(np.take(time_hod,mi), 0*np.take(time_hod,mi), *np.take(ae2lm(sun_ae[0], sun_ae[1], r.el_deg), mi, axis=1), scale=None, color='k', alpha=0.5)
                ax.quiver(np.take(time_hod,mi), 0*np.take(time_hod,mi), *np.take(ae2lm(wind_ae[0], wind_ae[1], r.el_deg), mi, axis=1), scale=None, color='k', alpha=0.5)
        ax.set_ylabel("Attack vector towards bore sight\n in tangent plane [x-El,El]"); ax.set_yticks([]); ax.legend(["Sun","Wind"]); ax.grid(True)
        # Feed offsets
        ax = axes[5*i+2]
        for f,r in zip(fs,rs):
            for fof,st in [(r.feedoffsetsH,'o'),(r.feedoffsetsV,'^')]:
                ax.plot(r.info["time_hod"], fof[f,:,0], 'C0'+st)
                ax.plot(r.info["time_hod"], fof[f,:,1], 'C1'+st)
                ax.plot(r.info["time_hod"], fof[f,:,2], 'C2'+st)
        if separate_freqs:
            ax.set_title("%.1fMHz"%(r.f_MHz[f]))
        ax.set_ylabel("Feed offsets [mm]\n%s"%lbl); ax.legend(["X_f H","Y_f H","Z_f H","X_f V","Y_f V","Z_f V"]); ax.grid(True)
        # Error beam
        _eb_ = 1 if (EBextra=="RMS") else 2
        ax = axes[5*i+3]
        for f,r,mi in zip(fs,rs,mask_ix):
            ax.errorbar(r.info["time_hod"], r.errbeamH[f,:,0]*100, yerr=r.errbeamH[f,:,3]*100, fmt='C0o') # max
            ax.errorbar(r.info["time_hod"], r.errbeamV[f,:,0]*100, yerr=r.errbeamV[f,:,3]*100, fmt='C1^')
            ax.plot(r.info["time_hod"], r.errbeamH[f,:,_eb_]*100, 'C0_') # RMS or stddev
            ax.plot(r.info["time_hod"], r.errbeamV[f,:,_eb_]*100, 'C1|')
            if (len(mi)>0): # Overplot the masked out points to give an idea how much EB is affected by masking
                ax.plot(np.take(r.info["time_hod"],mi), np.take(r.errbeamH.data[f,:,0]*100,mi), 'ko', alpha=0.3)
                ax.plot(np.take(r.info["time_hod"],mi), np.take(r.errbeamV.data[f,:,0]*100,mi), 'k^', alpha=0.3)
        if separate_freqs:
            ax.set_title("%.1fMHz"%(r.f_MHz[f]))
        ax.set_ylabel("Error beam (max & %s) [%%]\n%s"%(EBextra,lbl)); ax.legend(["H","V"]); ax.set_ylim(0,10); ax.grid(True)
        # Phase efficiency
        ax = axes[5*i+4]
        ax_ = ax.twinx()
        for f,r in zip(fs,rs):
            ax.plot(r.info["time_hod"], r.rpeffH[f,:,eff_ix], 'C0o') # H at spec freq
            ax.plot(r.info["time_hod"], r.rpeffV[f,:,eff_ix], 'C1^') # V at spec freq
            ax_.plot(r.info["time_hod"], r.rmsH[f,:], 'C0_') # H
            ax_.plot(r.info["time_hod"], r.rmsV[f,:], 'C1|') # V
        if separate_freqs:
            ax.set_title("%.1fMHz"%(r.f_MHz[f]))
        ax.legend(["H eff","V eff"], loc='upper left'); ax_.legend(["H RMS", "V RMS"], loc='upper right')
        ax.set_ylabel("Reflector phase eff @%gGHz [frac]\n%s"%(fspec_MHz[eff_ix]/1e3,lbl)); ax.grid(True)
        ax_.set_ylabel("Aperture deviation RMS [mm]", color="b")
    ax.set_xlabel("Hour of Day [local time]")
    ax.set_xlim(-0.01,24.01)


def _flatten_(twod, invmask=False): # Concatenates arrays (even zero-dimensional arrays) along the first axis
    ll = [np.atleast_1d(x) for x in twod]
    if invmask:
        ll = [np.ma.masked_where(~r.mask, r.data) if hasattr(r, "mask") else r for r in ll]
    return np.ma.concatenate(ll, axis=0)

def plot_errbeam_el(RS, labels, figsize=(14,4), extra="RMS"): 
    """ Generates a figure of error beam vs elevation angle
        @param RS: set of lists of 'HologResults'
        @param labels: a text label for each set of results """
    _eb_ = 1 if (extra=="RMS") else 2
    layout = _plan_layout_(RS, labels, separate_freqs=False)
    
    axes = np.atleast_1d(plt.subplots(len(layout),1,sharex=True,figsize=(figsize[0],figsize[1]*len(layout)))[1])
    for ax,(fs,rs,lbl) in zip(axes,layout):
        el = _flatten_([r.el_deg for r in rs])
        ax.errorbar(el, _flatten_([r.errbeamH[f,...,0]*100 for f,r in zip(fs,rs)]), yerr=_flatten_([r.errbeamH[f,...,3]*100 for f,r in zip(fs,rs)]), fmt='C0o', label="H") # max
        ax.plot(el, _flatten_([r.errbeamH[f,...,_eb_]*100 for f,r in zip(fs,rs)]), 'C0_') # RMS or stddev
        ax.errorbar(el, _flatten_([r.errbeamV[f,...,0]*100 for f,r in zip(fs,rs)]), yerr=_flatten_([r.errbeamV[f,...,3]*100 for f,r in zip(fs,rs)]), fmt='C1^', label="V")
        ax.plot(el, _flatten_([r.errbeamV[f,...,_eb_]*100 for f,r in zip(fs,rs)]), 'C1|')
        ax.set_ylabel("Error beam (max & %s) [%%]\n%s"%(extra,lbl)); ax.set_ylim(0,10); ax.grid(True)
    ax.set_xlabel("Elevation [deg]"); ax.legend()


def plot_offsets_el(RS, labels, figsize=(14,4), fit=None, hide=""):
    """ Generates a figure of feed offsets vs elevation angle 
        @param RS: set of lists of 'HologResults'
        @param labels: a text label for each set of results
        @param fit: e.g. 'lin' to generate linear fits for each of X, Y & Z (default None)
        @param hide: any subset of "XYZ", to hide the corresponding offset (default "") """
    layout = _plan_layout_(RS, labels, separate_freqs=False)
    
    axes = np.atleast_1d(plt.subplots(len(layout),1,sharex=True,figsize=(figsize[0],figsize[1]*len(layout)))[1])
    for ax,(fs,rs,lbl) in zip(axes,layout):
        fits = []
        el = _flatten_([r.el_deg for r in rs])
        for p,q in enumerate("XYZ"):
            if (q in hide): continue
            ax.plot(el, _flatten_([r.feedoffsetsH[f,...,p] for f,r in zip(fs,rs)]), 'C%do'%p, label="%s_f H"%q)
            ax.plot(el, _flatten_([r.feedoffsetsV[f,...,p] for f,r in zip(fs,rs)]), 'C%d^'%p, label="%s_f V"%q)
            if (fit != None) and (np.max(el)-np.min(el) > 15): # Fit offsets vs. elevation angle
                model = lambda offset, slope, el: offset + slope*el
                C = [[r.feedoffsetsH[f,p], r.feedoffsetsV[f,p]] for f,r in zip(fs,rs)]
                el, C = np.asarray(el), np.asarray(C)
                fitp = sop.fmin_powell(lambda p: np.nansum((C-np.stack([model(*p,el=el)]*2,-1))**2), [0,0], full_output=True, disp=False) # Same model fitted to both polarisations
                # Solid line if fitted without warnings
                ax.plot(np.sort(el), model(*fitp[0], el=np.sort(el)), ('C%d'%p) + ('-' if (fitp[-1]==0) else '--'), alpha=0.3)
                fits.append("%s_f=%.2f + %.2fEl"%(q, fitp[0][0], fitp[0][1]))
        if (len(fits) > 0):
            print("%s\t %s"%(lbl, ";".join([str(f) for f in fits])))
        ax.set_ylabel("Feed offsets [mm]\n%s"%lbl); ax.grid(True)
            
    ax.set_xlabel("Elevation [deg]"); ax.legend()


def plot_eff_el(RS, labels, fspec_MHz=(15000,20000), eff_ix=-1, figsize=(14,4)): 
    """ Generates a figure of phase efficiency (at max spec freq) vs elevation angle
        @param RS: set of lists of 'HologResults'
        @param labels: a text label for each set of results
        @param fspec_MHz: the frequencies corresponding to the last columns in phase_eff [MHz]
        @param eff_ix: selects which efficiency results to plot (as a negative index into 'fspec_MHz') (default -1) """
    N = len(labels)
    axes = np.atleast_1d(plt.subplots(N,1,sharex=True,figsize=(figsize[0],figsize[1]*N))[1])
    for ax,rs,lbl in zip(axes,RS,labels):
        el = _flatten_([r.el_deg for r in rs])
        ax.plot(el, _flatten_([np.min(r.rpeffH[...,eff_ix],axis=0) for r in rs]), 'C0o', label="H eff")
        ax.plot(el, _flatten_([np.min(r.rpeffV[...,eff_ix],axis=0) for r in rs]), 'C1^', label="V eff")
        ax.set_ylabel("Reflector phase eff @%gGHz [frac]\n%s"%(fspec_MHz[eff_ix]/1e3,lbl)); ax.grid(True)
        ax_ = ax.twinx()
        ax_.plot(el, _flatten_([np.max(r.rmsH,axis=0) for r in rs]), 'C0_', label="H RMS")
        ax_.plot(el, _flatten_([np.max(r.rmsV,axis=0) for r in rs]), 'C1|', label="V RMS")
        ax_.set_ylabel("Aperture deviation RMS [mm]", color="b")
    ax.set_xlabel("Elevation [deg]"); ax.legend(loc='upper left'); ax_.legend(loc='upper right')


def pad_rect(arrays, pad_value=np.nan):
    padded = []
    lengths = [len(np.atleast_1d(array)) for array in arrays]
    for a,l in zip(arrays,lengths):
        A = np.pad(np.atleast_1d(a), (0,max(lengths)-l), 'constant', constant_values=pad_value)
        padded.append(A)
    return padded
        
def plot_eff_freq(RS, labels, fspec_MHz=(15000,20000), figsize=(14,4)):
    """ Generates a figure of phase efficiency vs frequency.
        @param RS: set of lists of 'HologResults'
        @param labels: a text label for each set of results
        @param fspec_MHz: the frequencies corresponding to the last columns in phase_eff [MHz] """
    layout = _plan_layout_(RS, labels, separate_freqs=False)
    
    axes = np.atleast_1d(plt.subplots(len(layout),1,sharex=True,figsize=(figsize[0],figsize[1]*len(layout)))[1])
    for ax,(fs,rs,lbl) in zip(axes,layout):
        for ix,st,q in [('rpeffH','C0o','H'), ('rpeffV','C1^','V')]:
            ax.plot([r.f_MHz[f] for f,r in zip(fs,rs)], pad_rect([eval("r.%s"%ix)[f,...,0] for f,r in zip(fs,rs)]), st, label=q)
            for fi in range(len(fspec_MHz)):
                ax.plot([fspec_MHz[fi] for r in rs], pad_rect([eval("r.%s"%ix)[f,...,1+fi] for f,r in zip(fs,rs)]), st)
        ax.set_ylabel("Reflector phase eff [frac]\n%s"%lbl); ax.grid(True)
    ax.set_xlabel("Frequency [MHz]")#; ax.legend() # TODO: fix ordering above to avoid legend cloning


def plot_signalpathstats(rec, figsize=(14,16)):
    """ Plots status of overload & ADC RF power, for all selected antennas & all selected timestamps.
        @param rec: a ResultSet that's already beean loaded.
    """
    axes = plt.subplots(4,1, sharex=True, figsize=(figsize[0],figsize[1]))[1]
    # Plot some extra info from sensor logs
    dataset = np.atleast_1d(rec.beams[0])[0].dataset
    T = dataset.rawtime
    band = dataset.band[0].lower() # Dataset bands (e.g. UHF, L, S) -> sensor naming ("u", "l", "s")
    for ant in dataset.radialscan_allantenna:
        for i,P in enumerate(["H","V"]):
            v = katselib.getsensorvalues("%s_dig_%s_band_rfcu_%spol_overload"%(ant,band,P.lower()), T)[1]
            axes[0].plot(T-T[0], v, '_|'[i], label="%s %s-pol"%(ant,P))
            axes[0].set_ylabel("RFCU overloaded"); axes[0].set_ylim(-0.1,1.1); axes[0].legend()
            v = katselib.getsensorvalues("%s_dig_%s_band_adc_%spol_rf_power_in"%(ant,band,P.lower()), T)[1]
            axes[1].plot(T-T[0], v, ["-","--"][i], label="%s %s-pol"%(ant,P))
            axes[1].set_ylabel("ADC RF input power [dBm]"); axes[1].grid(True); axes[1].legend()
    
    # Plot boresight visibilities
    labels = [] # Collect labels to avoid repetition if there's more than one cycle
    for freq_MHz,beams in zip(rec.f_MHz, rec.beams): # The desired data is attached to the 'beams'
        for beam in np.atleast_1d(beams):
            ts, rb, prod = beam.rawonboresight
            for p,x in enumerate(rb):
                for ax,quant in zip(axes[-2:], [np.abs(x), np.unwrap(np.angle(x))*180/np.pi]):
                    ax.plot(ts-T[0], quant, 'C%d.'%((p+len(labels))%10)) # CN must have N 0..9
        labels.extend(["%s @ %.3fGHz"%(p,freq_MHz/1e3) for p in prod])
    for ax in axes[-2:]:
        ax.legend(labels)
        if (len(np.atleast_1d(beams)) > 1): # Add markers to identify multiple cycles
            for beam in beams:
                ts = beam.rawonboresight[0]
                ax.axvline(ts[0]-T[0], color='k', alpha=0.1)
    axes[-2].set_ylabel("Raw visibilities amplitude [linear]"); axes[-2].grid(True)
    axes[-1].set_ylabel("Raw visibilities phase [deg]"); axes[-1].grid(True)
    axes[-1].set_xlabel("Time [sec]")


def snr_mask(beams, SNR_min=30, phaseRMS_max=30):
    """ Calculate SNR metrics & generate a mask that may be used to flag out poor quality data. All cycles for which
        the bore sight phase scatter relative to a piecewise linear fit (8 segments) exceeds 'phaseRMS_max' OR where
        bore sight amplitude scatter (mean-to-stddev ratio) is below 'SNR_min', are flagged in the returned mask.
        Only the co-polarisation data is assessed since the cross-pol is typically only significant for circular pol
        @param beams: a set of ResultSet.beams as loaded with load_data(), for a specific frequency i.e. dimension either 0 or 1D.
        @param SNR_min: a cycle for which the amplitude SNR (mean/stddev) is below this threshold is masked as 'True' (default 30).
        @param phaseRMS_max: a cycle for which the phase scatter (relative to piecewise linear) is above this threshold is masked as 'True' (default 30).
        @return: (SNR, mask) with SNR as (cycles,[[SNR_p0_amplitude,std_p0_phase],...]) & mask as (cycles)
    """
    snr = [] # per cycle ([SNR_p0_amplitude,std_p0_phase],...)
    mask = [] # per cycle
    for beam in np.atleast_1d(beams): # Cycles
        x, prods = beam.rawonboresight[1:] # x is visibilities (products,time)
        prods = [p.split("-") for p in prods] # Convert from "xx-yy" to (xx,yy)
        ix = [i for i in range(len(prods)) if (prods[i][0][-1]==prods[i][1][-1])]
        a, ph = np.abs(x), np.unwrap(np.angle(x))*180/np.pi
        bp = np.linspace(0, len(x[0]), 8, dtype=int) # Breakpoint indices for piecewise linear fits; expect issues with fewer than 4 & more than 10 segments!
        a_, ph_ = sig.detrend(a, bp=bp), sig.detrend(ph, bp=bp)
        snr.append(zip(np.mean(a,axis=1)/np.std(a_,axis=1), np.std(ph_,axis=1))) # count/count, deg, arranged as (product)
        snr_a, rms_ph = np.transpose(np.min(np.asarray(snr[-1])[ix,:], axis=0)) # Use only co-pol for the mask
        mask.append((snr_a<SNR_min) or (rms_ph>phaseRMS_max))
    return np.asarray(snr), np.asarray(mask)


def HOD(rawtime, tzoffset):
    """ @param rawtime: raw timestamp ito UTC.
        @param tzoffset: time zone offset [hrs] for the telescope.
        @return: hour of day in local time [0,24] with timezone offset from UTC as specified. 
    """
    tgm = time.gmtime(rawtime) # Don't use time.localtime() since servers are sometimes at UTC
    return ((tgm[3] + tgm[4]/60.0 + tgm[5]/3600.0 + tzoffset) % 24)


def geterrorbeam(measuredG, modelG, crop=0, contourdB=-20, centered=True): # Minor elaboration from katholog.hologreport.geterrorbeam()
    """ Computes the errorr beam as measuredG_n**2-modelG_n**2, where _n implies normalization to the peak amplitude.
    
        @param meaduredG, modelG: far field amplitude maps, centred on identical grids.
        @param crop: the fraction by which, if > 0, measuredG has to be cropped to match modelG, or vice versa if < 0  (default 0).
        @param contourdB: error beam below this level (on modelG_n) is set to NaN (default -20).
        @param centered: False to refine the centering of the measured map after zoom (default True).
        @return: errorbeam map
    """
    # Rescale without re-centering
    measuredG = katsemat.cropped_zoom(np.abs(measuredG), crop if (crop>0) else 0, 0,0)
    modelG = katsemat.cropped_zoom(np.abs(modelG), 0 if (crop>0) else -crop, 0,0)
    
    if not centered: # Re-centre, BUT experimentation suggests this is always worse than assuming the maps were centred to begin with
        shape = modelG.shape
        centre = np.mean(np.argwhere(modelG>=0.7*np.nanmax(modelG)), axis=0) # y,x
        modelG = katsemat.cropped_zoom(modelG, 1, centre[1]-shape[1]/2., centre[0]-shape[0]/2.)
        print("zoomed ", 0 if (crop>0) else -crop, centre[1]-shape[1]/2., centre[0]-shape[0]/2.)

        centre = np.mean(np.argwhere(measuredG>=0.7*np.nanmax(measuredG)), axis=0) # y,x
        measuredG = katsemat.cropped_zoom(measuredG, 1, centre[1]-shape[1]/2., centre[0]-shape[0]/2.)
        print("zoomed ", crop if (crop>0) else 0, centre[1]-shape[1]/2., centre[0]-shape[0]/2.)
    
    # Normalize both patterns & generate the error pattern
    measuredG = measuredG/np.nanmax(measuredG)
    modelG = modelG/np.nanmax(modelG)
    errorbeam = measuredG**2 - modelG**2
    
    # Mask out regions to be ignored
    errorbeam[20*np.log10(modelG) < contourdB] = np.nan
#     errorbeam[20*np.log10(measuredG) > -0.01] = np.nan # Some error patterns have a spike at bore sight, an artefact from fitpoly?
    
    return errorbeam

def standard_report(measured, predicted=None, DF=5, spec_freq_MHz=[15000,20000], tzoffset=2, contourdB=-20, beampolydegree=28, beamsmoothing="zernike", eb_extent=(-0.2,0.2), coords="SKA", debug=False, makepdf=True, pdfprefix="", **devkwargs):
    """ Makes standard plots and prints information for the supplied holography result sets.
        
        @param measured: the result set to report on [ResultSet]
        @param predicted: predicted patterns to de-embed from measured (default None) [ResultSet]
        @param DF: max frequency offset to an acceptable predicted pattern [MHz] (default 5)
        @param tzoffset: time zone offset of observatory relative to timestamps (default +2).
        @param contourdB: contour within which to determine the 'error beam', in dB below peak (default -20).
        @param beampolydegree: degree of polynomial to smooth the beams with (default 28 for zernike out to radius ~7*HPBW; increase by 2 for each additional HPBW added to radius).
        @param beamsmoothing: "zernike" (default) to smooth the beam using zernike polynomials, anything else for regular polynomial fitting.
        @param coords: "SKA" for feed offsets in SKA coordinate system, instead of the standard katholog convention (default "SKA")
        @param devkwargs: passed to apmap.plot('dev') e.g. 'clim'. May also include 'cmap' to override
                          the colormap for 'dev' maps only.
        @return: el [deg], freq [MHz], feedoffsetsH_f [mm], feedoffsetsV_f [mm], refl_phase_effH [frac], refl_phase_effV [frac],
                 rmsH [mm], rmsV [mm], errbeamH [frac], errbeamV [frac], info {time_hod,deg_per_sec,feedindexer_deg,enviro:{sun_deg,sun_rel_deg,temp_C,wind_mps,wind_rel_deg,}}
                 Note: feedoffsets are vs freq, refl_phase_eff (as-is) are vs (freq,freq+spec_freqs), rms (as-is) are vs freq,
                       errbeam are vs (freq,[max,95pct,stddev]).
    """
    devcmap = devkwargs.pop('cmap', None)
    
    feedoffsetsH, feedoffsetsV, refl_phase_effH, refl_phase_effV, rmsH, rmsV = [], [], [], [], [], []
    errbeamH, errbeamV, el_deg, time_hod, deg_per_sec, feedindexer_deg, enviro = [], [], [], [], [], [], []
    
    b0 = np.atleast_1d(measured.beams[0])[0] # Handles cycles & singles
    key = (measured.fid, b0.scanantennaname, b0.trackantennanames[0])
    pp = katselib.PDFReport("%s_hologreport_%s_%d.pdf"%(pdfprefix,key[1],key[0]), header="%d: %s referenced to %s"%key, pagesize=(11,17), save=makepdf)
    try:
        pp.capture_stdout(echo=True)
        print("Target: %s"%b0.dataset.target.name)
        print("Processing tags: %s"%measured.tags)
        
        for f_MHz, beacon_pol, beams, apmapsH, apmapsV in zip(measured.f_MHz, measured.beacon_pol, measured.beams, measured.apmapsH, measured.apmapsV):
            print("\n"+ "-"*20 + " %.1fMHz"%f_MHz)
            
            feedoffsetsH.append([]); feedoffsetsV.append([]); refl_phase_effH.append([]); refl_phase_effV.append([]); rmsH.append([]); rmsV.append([])
            errbeamH.append([]); errbeamV.append([]); el_deg.append([]); time_hod.append([]); deg_per_sec.append([])
            feedindexer_deg.append([]); enviro.append([])
            
            _predicted_ = None
            if (predicted is not None):
                _predicted_ = [(abs(f-f_MHz),f,p,b,h,v) for f,p,b,h,v in zip(predicted.f_MHz,predicted.beacon_pol,predicted.beams,predicted.apmapsH,predicted.apmapsV) if (p==beacon_pol) and (abs(f-f_MHz)<=DF)]
                _predicted_ = None if (len(_predicted_) == 0) else sorted(_predicted_)[0] # Sorted by df, ascending
            
            snr = snr_mask(beams)[0] # (cycles,products)
            for ci,(beam,apmapH,apmapV) in enumerate(zip(np.atleast_1d(beams),np.atleast_1d(apmapsH),np.atleast_1d(apmapsV))): # Walk through each measurement in the set
                el_deg[-1].append(beam.el_deg)
                time_hod[-1].append(HOD(beam.time_avg, tzoffset=tzoffset))
                deg_per_sec[-1].append(beam.deg_per_sec)
                feedindexer_deg[-1].append(beam.feedindexer_deg)
                enviro[-1].append(dict(sun_deg=beam.sun_deg, sun_rel_deg=beam.sun_rel_deg, temp_C=beam.temp_C, wind_mps=beam.wind_mps, wind_rel_deg=beam.wind_rel_deg))
                print(">> %.1f degEl @ %.2f hrs [local time]; SNR~%s"%(el_deg[-1][-1], time_hod[-1][-1], np.array2string(snr[ci],precision=0).replace("\n",",")))
                
                _apmapH, _apmapV = apmapH, apmapV # Un-modified copies, in case they get modified below
                # If predicted maps provided, generate copies of measured maps that are corrected by predicted maps
                if (_predicted_ is not None):
                    p_apmapH, p_apmapV = _predicted_[-2:]
                    scale = apmapH.freqMHz/p_apmapH.freqMHz # First order scaling, if DF is small
                    apmapH = re_analyse(apmapH, feedphasemap=p_apmapH.unwrappedphasemap*scale) # Takes care of aperture phase AND squint
                    apmapV = re_analyse(apmapV, feedphasemap=p_apmapV.unwrappedphasemap*scale)
                
                if (ci == 0): # Only figures for the first cycle
                    plt.figure(figsize=(14,18))
                    plt.suptitle("%.1fMHz @ %.1fdegEl, %.1fhrs [local time]"%(f_MHz,el_deg[-1][-1],time_hod[-1][-1]) +
                                 "\nAs-is")
                    plt.subplot(4,2,1); beam.plot("Gx", doclf=False); plt.title("Gx")
                    if (_predicted_ is not None): plt.contour(_predicted_[-3].margin, _predicted_[-3].margin, 20*np.log10(np.abs(_predicted_[-3].Gx[0])), [-43], alpha=0.2)
                    plt.subplot(4,2,2); beam.plot("Dx", doclf=False); plt.title("Dx")
                    plt.subplot(4,2,3); beam.plot("Dy", doclf=False); plt.title("Dy")
                    plt.subplot(4,2,4); beam.plot("Gy", doclf=False); plt.title("Gy")
                    if (_predicted_ is not None): plt.contour(_predicted_[-3].margin, _predicted_[-3].margin, 20*np.log10(np.abs(_predicted_[-3].Gy[0])), [-43], alpha=0.2)
                    plt.subplot(4,2,5); _apmapH.plot('amp', doclf=False); plt.title("Aperture amplitude x")
                    plt.subplot(4,2,6); _apmapV.plot('amp', doclf=False); plt.title("Aperture amplitude y")
                    if (devcmap is not None): # Adjust this only for the devmaps
                        _CM_ = apmapH.colmap
                        apmapH.colmap = apmapV.colmap = _apmapH.colmap = _apmapV.colmap = devcmap
                    plt.subplot(4,2,7); _apmapH.plot('nopointingdev', doclf=False, **devkwargs); plt.title("Aperture path length deviation x") # Just pointing removed
                    plt.subplot(4,2,8); _apmapV.plot('nopointingdev', doclf=False, **devkwargs); plt.title("Aperture path length deviation y")
                    pp.report_fig(max(plt.get_fignums()))
                
                    
                    N = 2 if (_predicted_ is None) else 3
                    plt.figure(figsize=(14,14+(N-2)*4))
                    plt.suptitle("%.1fMHz @ %.1fdegEl, %.1fhrs [local time]"%(f_MHz,el_deg[-1][-1],time_hod[-1][-1]) +
                                 "\nAperture plane deviations, corrected for pointing error")
                    plt.subplot(N,2,1); _apmapH.plot('nopointingdev', doclf=False, **devkwargs)
                    plt.subplot(N,2,2); _apmapV.plot('nopointingdev', doclf=False, **devkwargs)
                    # If predicted maps provided, generate devmaps with those subtracted
                    if (_predicted_ is not None):
                        plt.subplot(N,2,3); apmapH.plot('nopointingdev', doclf=False, **devkwargs); plt.title("H: Feed Phase removed")
                        plt.subplot(N,2,4); apmapV.plot('nopointingdev', doclf=False, **devkwargs); plt.title("V: Feed Phase removed")
                        TAG = "Feed Phase removed & re-collimated"
                    else:
                        TAG = "Re-collimated"
                    plt.subplot(N,2,2*N-1); apmapH.plot('dev', doclf=False, **devkwargs); plt.title("H: %s"%TAG)
                    if debug: # Extend devmap beyond blanked-out dishdiameter
                        plt.imshow(katholog.utilities.getdeviation(apmapH.nopointingphasemap,apmapH.mapsize,apmapH.gridsize,apmapH.wavelength,apmapH.focallength,apmapH.parabolaoffsetdev),
                                   cmap=apmapH.colmap, extent=[i*apmapH.mapsize/2. for i in [-1,1,-1,1]], origin='lower', **devkwargs)
                    plt.subplot(N,2,2*N); apmapV.plot('dev', doclf=False, **devkwargs); plt.title("V: %s"%TAG)
                    if debug: # Extend devmap beyond blanked-out dishdiameter
                        plt.imshow(katholog.utilities.getdeviation(apmapV.nopointingphasemap,apmapV.mapsize,apmapV.gridsize,apmapV.wavelength,apmapV.focallength,apmapV.parabolaoffsetdev),
                                   cmap=apmapV.colmap, extent=[i*apmapV.mapsize/2. for i in [-1,1,-1,1]], origin='lower', **devkwargs)
                    pp.report_fig(max(plt.get_fignums()))
                    
                    if (devcmap is not None): # Restore state
                        apmapH.colmap = apmapV.colmap = _apmapH.colmap = _apmapV.colmap = _CM_
                
                
                TAG = "" if (_predicted_ is None) else "(feed removed)"
                
                feedoffsetH, feedoffsetV = apmapH.feedoffset, apmapV.feedoffset
                if (coords == "SKA"): # katholog -> SKA coordinates
                    feedoffsetH, feedoffsetV = [1,-1,1]*np.take(feedoffsetH,(1,0,2)), [1,-1,1]*np.take(feedoffsetV,(1,0,2))
                feedoffsetsH[-1].append(feedoffsetH)
                feedoffsetsV[-1].append(feedoffsetV)
                print("%s Feed XYZ_f offsets %s [mm]"%(coords,TAG))
                print("    H-pol %s "%(feedoffsetsH[-1][-1]) + "\t\t" + "V-pol %s"%(feedoffsetsV[-1][-1]) + "\t(FI @ %g deg)"%feedindexer_deg[-1][-1][0])
                if debug:
                    apmapH.printoffset(); print("")
                    apmapV.printoffset(); print("")
                
                rmsH[-1].append(apmapH.rms0_mm)
                rmsV[-1].append(apmapV.rms0_mm)
                print("Aperture plane RMS %s [mm]:  as-is; re-collimated"%TAG)
                print("              H-pol %.2f; %.2f\t\tV-pol %.2f; %.2f"%(rmsH[-1][-1], apmapH.rms_mm, rmsV[-1][-1], apmapV.rms_mm))
                
                print("Reflector phase efficiency %s:  as-is; re-collimated"%TAG)
                refl_phase_effH[-1].append([])
                refl_phase_effV[-1].append([])
                for f in [f_MHz*1e6]+[sf*1e6 for sf in spec_freq_MHz]:
                    _apmapH.gain(freqscaling=f/apmapH.freq)
                    apmapH.gain(freqscaling=f/apmapH.freq)
                    refl_phase_effH[-1][-1].append(apmapH.eff0_phase)
                    sH = "H-pol %g; %g"%(refl_phase_effH[-1][-1][-1], apmapH.eff_phase)
                    _apmapV.gain(freqscaling=f/apmapV.freq)
                    apmapV.gain(freqscaling=f/apmapV.freq)
                    refl_phase_effV[-1][-1].append(apmapV.eff0_phase)
                    sV = "V-pol %g; %g"%(refl_phase_effV[-1][-1][-1], apmapV.eff_phase)
                    print("    %.1f GHz: "%(f/1e9) + sH + "\t\t" + sV)
                if debug:
                    apmapH.printgain(); print("")
                    apmapV.printgain(); print("")

                # Error beam within specified contour
                if (_predicted_ is not None):
                    p_beam = _predicted_[-3]
                    smoothbeam(beam, fitdBlevel=contourdB-3, degree=(p_beam.Gx[0],p_beam.Gy[0]) if (beamsmoothing=='fourier') else beampolydegree, kind=beamsmoothing)
                    if (ci == 0):
                        smoothbeam(p_beam, fitdBlevel=contourdB-3, degree=(p_beam.Gx[0],p_beam.Gy[0]) if (beamsmoothing=='fourier') else beampolydegree, kind=beamsmoothing)
                        # Only figures for the first cycle
                        axes = plt.subplots(1,2, figsize=(14,8))[1]
                        plt.suptitle("%.1fMHz @ %.1fdegEl, %.1fhrs [local time]"%(f_MHz,el_deg[-1][-1],time_hod[-1][-1]) +
                                     "\nError Beam")
                    sHV = []
                    resid = lambda G,mG: np.nanstd(np.abs(G)-np.abs(mG))
                    for ax,(lbl,meas,sigma,modl,res) in zip(axes,[("H",beam.mGx[0],resid(beam.Gx[0],beam.mGx[0]),p_beam.mGx[0],errbeamH[-1]),
                                                                  ("V",beam.mGy[0],resid(beam.Gy[0],beam.mGy[0]),p_beam.mGy[0],errbeamV[-1])]):
                        ext_ = lambda bm: bm.extent/(300.0/bm.freqgrid[0]) # Normalized to HPBW*D
                        crop = ext_(beam)/ext_(p_beam) - 1 # Measured to be cropped by this fraction if > 0
                        errorbeam = geterrorbeam(meas, modl, crop, contourdB=contourdB)
                        max_eb, fs_eb, std_eb = np.nanmax(np.abs(errorbeam)), np.nanpercentile(np.abs(errorbeam),95), np.nanstd(errorbeam)
                        nn_eb = sigma/np.nanmax(np.abs(meas)) # 1sigma measurement noise in the same scale as errorbeam
                        sHV.append("%s-pol < %.1f[%.1f]%% (95pct %.1f%%, std %.1f%%)"%(lbl, max_eb*100, nn_eb*100, fs_eb*100, std_eb*100))
                        res.append([max_eb, fs_eb, std_eb, nn_eb])
                        if (ci == 0): # Only figures for the first cycle
                            im = ax.imshow(errorbeam, origin='lower', extent=[min(beam.extent,p_beam.extent)/2.*i for i in [-1,1,-1,1]]) # Square
                            ax.contour(p_beam.margin, p_beam.margin, 20*np.log10(np.abs(modl)), [contourdB], alpha=0.2) # Especially useful to see if smoothing broadens the pattern
                            ax.set_title("Error Beam %s-pol [frac]"%lbl)
                            ax.set_xlabel("degrees"); ax.set_ylabel("degrees")
                            ax.set_xlim(*eb_extent); ax.set_ylim(*eb_extent)
                    if (ci == 0): # Only figures for the first cycle
                        plt.colorbar(im, ax=axes)
                        pp.report_fig(max(plt.get_fignums()))
                    print("Error Beam")
                    print("    %s\t\t%s"%(sHV[0],sHV[1]))
                else:
                    errbeamH[-1].append([np.nan, np.nan, np.nan, np.nan])
                    errbeamV[-1].append([np.nan, np.nan, np.nan, np.nan])
            
            # Reference patterns for this frequency, if any
            if (_predicted_ is not None):
                beam,apmapH,apmapV = _predicted_[-3:]
                plt.figure(figsize=(14,18))
                plt.suptitle("Reference patterns @ %.1fMHz"%_predicted_[1])
                plt.subplot(4,2,1); beam.plot("Gx", doclf=False); plt.xlim(-beam.extent/2., beam.extent/2.); plt.ylim(-beam.extent/2., beam.extent/2.); plt.title("Gx")
                plt.subplot(4,2,2); beam.plot("Dx", doclf=False); plt.xlim(-beam.extent/2., beam.extent/2.); plt.ylim(-beam.extent/2., beam.extent/2.); plt.title("Dx")
                plt.subplot(4,2,3); beam.plot("Dy", doclf=False); plt.xlim(-beam.extent/2., beam.extent/2.); plt.ylim(-beam.extent/2., beam.extent/2.); plt.title("Dy")
                plt.subplot(4,2,4); beam.plot("Gy", doclf=False); plt.xlim(-beam.extent/2., beam.extent/2.); plt.ylim(-beam.extent/2., beam.extent/2.); plt.title("Gy")
                plt.subplot(4,2,5); apmapH.plot('amp', doclf=False); plt.title("Aperture amplitude x")
                plt.subplot(4,2,6); apmapV.plot('amp', doclf=False); plt.title("Aperture amplitude y")
                plt.subplot(4,2,7); apmapH.plot('nopointingdev', doclf=False, **devkwargs); plt.title("Aperture path length deviation x")
                plt.subplot(4,2,8); apmapV.plot('nopointingdev', doclf=False, **devkwargs); plt.title("Aperture path length deviation y")
                pp.report_fig(max(plt.get_fignums()))
            
            if (ci == 0): # Only a single cycle, so reduce the len_1 lists to scalar values
                feedoffsetsH[-1] = feedoffsetsH[-1][0]; feedoffsetsV[-1] = feedoffsetsV[-1][0]; rmsH[-1] = rmsH[-1][0]; rmsV[-1] = rmsV[-1][0]
                refl_phase_effH[-1] = refl_phase_effH[-1][0]; refl_phase_effV[-1] = refl_phase_effV[-1][0]
                errbeamH[-1] = errbeamH[-1][0]; errbeamV[-1] = errbeamV[-1][0]; feedindexer_deg[-1] = feedindexer_deg[-1][0]
                el_deg[-1] = el_deg[-1][0]; time_hod[-1] = time_hod[-1][0]; deg_per_sec[-1] = deg_per_sec[-1][0]; enviro[-1] = enviro[-1][0]

        # Reduce the dimensions that don't vary with frequency
        feedindexer_deg = feedindexer_deg[0]; el_deg = el_deg[0]; time_hod = time_hod[0]; deg_per_sec = deg_per_sec[0]; enviro = enviro[0]
        el_deg = el_deg if (ci == 0) else np.asarray(el_deg) # Make it easy to work with data from multiple cycles
        
        pp.report_stdout()
        
        # Signal path statistics
        plot_signalpathstats(measured)
        plt.suptitle("Signal Path Statistics")
        pp.report_fig(max(plt.get_fignums()))
        
        results = HologResults(el_deg, measured.f_MHz, np.asarray(feedoffsetsH), np.asarray(feedoffsetsV),
                               np.asarray(refl_phase_effH), np.asarray(refl_phase_effV),
                               np.asarray(rmsH), np.asarray(rmsV), np.asarray(errbeamH), np.asarray(errbeamV),
                               dict(time_hod=time_hod,deg_per_sec=deg_per_sec,feedindexer_deg=feedindexer_deg,enviro=enviro))
        
        if (ci > 0): # Multiple cycles
            plot_vs_hod([[results]], ["%d"%measured.fid], False, spec_freq_MHz, figsize=(14,4))
            pp.report_fig(max(plt.get_fignums()))
            plot_errbeam_cycles([measured], predicted, DF=DF, beampolydegree=None, contourdB=contourdB, extent=eb_extent, ncols=6,
                                clim=[i*np.nanmedian(results.errbeamH[:,0]) for i in [-1,1]])
            pp.report_fig(max(plt.get_fignums()))
    finally:
        pp.close()
    
    return results


def plot_errbeam_cycles(recs, predicted, DF=5, beampolydegree=28, beamsmoothing="zernike", contourdB=-20, tzoffset=2, extent=(-0.2,0.2), clim=(-0.1,0.1), ncols=6, cmap=None):
    """ Plots H & V-pol error beams (not 'absolute value') for multi-cycle records.
        @param recs: list of 'ResultSet'
        @param predicted: predicted patterns to de-embed from measured (default None) [ResultSet]
        @param DF: max frequency offset to an acceptable predicted pattern [MHz] (default 5).
        @param tzoffset: time zone offset of observatory relative to timestamps (default +2).
        @param beampolydegree: degree of polynomial to smooth the beams with, or None if already smoothed (default 28, for zernike out to radius ~7*HPBW).
        @param beamsmoothing: "zernike" (default) to smooth the beam using zernike polynomials, anything else for regular polynomial fitting.
        @param contourdB: contour within which to determine the 'error beam', in dB below peak (default -20). """
    freqs = sorted(set(np.concatenate([rec.f_MHz for rec in recs])))
    for f_MHz in freqs: # Group figures by frequency
        for rec in recs:
            try:
                beams = np.atleast_1d(rec.beams[list(rec.f_MHz).index(f_MHz)]) # The beams at this frequency
            except ValueError: # This record doesn't cover the current frequency
                continue
            _predicted_ = [(abs(f-f_MHz),f,p,b) for f,p,b in zip(predicted.f_MHz,predicted.beacon_pol,predicted.beams) if (p==rec.beacon_pol[0]) and (abs(f-f_MHz)<=DF)]
            pbm = sorted(_predicted_)[0][-1] # Sorted by df, ascending
            if (beampolydegree):
                smoothbeam(pbm, fitdBlevel=contourdB-3, degree=(pbm.Gx[0],pbm.Gy[0]) if (beamsmoothing=='fourier') else beampolydegree, kind=beamsmoothing)
            C = min(len(beams), ncols)
            R = int(len(beams)/float(ncols) + 0.9999) # Columns & rows per pol
            axes = plt.subplots(2*R, ncols, figsize=(18,2*R*18./ncols))[1]
            plt.suptitle("%d\n %.1f MHz @ %gdegEl"%(rec.fid, f_MHz, beams[0].el_deg))
            for pol in [0,1]:
                try: # There is typically more axes per pol than there are cycles
                    for r in range(R):
                        for c in range(C):
                            ax = axes[r+pol*R][c]
                            if (pol == 0): ax.set_title("%.1f [LT]"%HOD(beams[r*C+c].time_avg, tzoffset=tzoffset))
                            if (c == 0): ax.set_ylabel("%s-pol"%("HV"[pol]))
                            bm = beams[r*C+c] # Just for first frequency
                            if (beampolydegree):
                                smoothbeam(bm, fitdBlevel=contourdB-3, degree=(pbm.Gx[0],pbm.Gy[0]) if (beamsmoothing=='fourier') else beampolydegree, kind=beamsmoothing)
                            meas, modl = (bm.mGx[0], pbm.mGx[0]) if (pol == 0) else (bm.mGy[0], pbm.mGy[0])
                            ext_ = lambda bm: bm.extent/(300/bm.freqgrid[0]) # Normalized to HPBW*D
                            crop = ext_(bm)/ext_(pbm) - 1 # Measured to be cropped by this fraction if > 0
                            eb = geterrorbeam(meas, modl, crop, contourdB=contourdB)
                            ax.imshow(eb, origin='lower', extent=[min(bm.extent,pbm.extent)/2.*i for i in [-1,1,-1,1]], clim=clim, cmap=cmap)
                            ax.set_xlim(*extent); ax.set_ylim(*extent)
                except IndexError: # End of cycles for this pol, continue to next
                    continue


def plot_enviro(recs, label, what="sun,wind,temp,humidity", tzoffset=0, figsize=(14,3)):
    """ Generates a figure of the following metrics vs. hour of day for measurements (singles & cycles):
        * sun & wind attack angles
        * wind speed
        * ambient temperature
        * relative humidity
        
        @param recs: list of 'ResultSet'
        @param what: a comma-separated subset of {wind, temp, humidity, elevation} """
    what = what.split(",")
    time_range = [np.nan, np.nan]
    axes = plt.subplots(len(what), 1, sharex=True, figsize=(figsize[0],figsize[1]*len(what)))[1]
    for r in recs:
        beams = np.atleast_1d(r.beams[0])
        time_avg = [bm.time_avg for bm in beams] # UTC
        time_hod = np.asarray([HOD(t, tzoffset) for t in time_avg]) # local time
        wrp = 0 if (len(time_hod)==1) else np.argmin(np.sign(np.diff(time_hod))) # Index position where time crossed over midnight
        wrp = [slice(0,None)] if (wrp == 0) else [slice(0,wrp+1), slice(wrp+1,None)]
        
        for ax,key in enumerate(what):
            if ("wind" in key): # Wind speed
                axes[ax].plot(time_hod, [bm.wind_mps[0] for bm in beams], 'b_') # avg
                for ix in wrp:
                    axes[ax].fill_between(time_hod[ix], [bm.wind_mps[1] for bm in beams[ix]], [bm.wind_mps[2] for bm in beams[ix]], facecolor='b', alpha=0.1)
                axes[ax].set_ylabel("Max avg wind speed [m/s]")
            elif ("elev" in key): # Sun & wind attack angle centred on boresight elevation angle
                el_deg = np.asarray([bm.el_deg for bm in np.atleast_1d(r.beams[0])])
                # Sun & wind angles are from boresight to sun & to wind origin, but attack vector is from those to boresight so change sign of l & m
                ae2lm = lambda Daz,Del,el: -np.asarray(katpoint._projection.sphere_to_ortho(az0=0,el0=el*np.pi/180,az=Daz*np.pi/180,el=(el+Del)*np.pi/180)[:2]) # deg,deg,deg -> -l,-m in tangent plane on bore sight
                sun_ae = np.transpose([np.asarray(bm.sun_rel_deg) for bm in np.atleast_1d(r.beams[0])]) # dAz,dEl from bore to sun
                axes[ax].quiver(time_hod, el_deg, *ae2lm(sun_ae[0], sun_ae[1], el_deg), scale=None, color='r', alpha=0.5)
                windaz_deg = np.asarray([bm.wind_rel_deg for bm in np.atleast_1d(r.beams[0])]) # dAz from bore sight to wind origin
                axes[ax].quiver(time_hod, el_deg, *ae2lm(windaz_deg, 0-el_deg, el_deg), scale=None, color='b', alpha=0.5)
                axes[ax].set_ylabel("Elevation angle [deg]")
                axes[ax].legend(["Sun","Wind"]); 
                axes[ax].set_title("Attack vector towards bore sight in [x-El,El] plane")
            elif ("temp" in key): # Ambient temperature
                axes[ax].plot(time_hod, [bm.temp_C[0] for bm in beams], 'r_') # avg
                for ix in wrp:
                    axes[ax].fill_between(time_hod[ix], [bm.temp_C[1] for bm in beams[ix]], [bm.temp_C[2] for bm in beams[ix]], facecolor='r', alpha=0.1)
                axes[ax].set_ylabel("Ambient temperature [degC]")
            elif ("humid" in key): # Humidity
                humidity = katselib.getsensorvalues("%s_enviro_air_relative_humidity"%"s0000", np.reshape([np.linspace(t-10*60,t+10*60,5) for t in time_avg],(-1,)))[1] # 20 min window
                humidity = np.reshape(humidity, (-1,5)) # 5 time points per cycle as per 'get' above
                humidity = (np.mean(humidity,axis=1), np.min(humidity,axis=1), np.max(humidity,axis=1))
                axes[ax].plot(time_hod, humidity[0], 'k_') # avg
                axes[ax].set_ylabel("Humidity [%%]"); axes[ax].grid(True)
            axes[ax].grid(True)
        
        time_range = [np.nanmin([time_range[0], np.min(time_avg)]), np.nanmax([time_range[-1], np.max(time_avg)])]
    # Last axis get the x-values
    axes[ax].set_xlabel("Hour of Day [local time]")
    axes[ax].set_xlim(-0.01,24.01)
    # First axis gets the title
    label = "" if (label is None) else label
    axes[0].set_title("%s\n%s - %s [local time]"%(label, time.ctime(time_range[0]+tzoffset*3600), time.ctime(time_range[-1]+tzoffset*3600)))


def plot_apmapdiffs(apmap0, apmap1, title, what="nopointingphasemap", vlim=None, masked=True):
    """ Generates a figure to show the differences between aperture plane maps.
        @param apmap0, apmap1: instances of katholog.ApertureMap
        @param what: the attribute of an ApertureMap to represent (default 'nopointingphasemap')
        @param vlim: limit the range of values of 'what' displayed, either None or (min,max) (default None).
        @return: the figure's axes (always a 2D list)
    """
    apmaps0 = np.atleast_1d(apmap0)
    apmaps1 = np.atleast_1d(apmap1)
    fig, axs = plt.subplots(len(apmaps0),2, figsize=(6*2,5*len(apmaps0)))
    axs = [axs] if (len(np.shape(axs))==1) else axs # Undo auto squeeze
    fig.suptitle("%s [%s]"%(title, what))
    unit = "mm" if ("dev" in what) else ("rad" if "phase" in what else "ampl")
    
    for ax_,apmap0,apmap1 in zip(axs,apmaps0,apmaps1):
        ax_[1].set_title("%.f - %.f"%(apmap0.dataset.env_times[1], apmap1.dataset.env_times[1]), x=0)
        diff = apmap0.__getattribute__(what) - apmap1.__getattribute__(what)
        if masked:
            diff[apmap0.maskmap * apmap1.maskmap == 1] = np.nan
        vlim = (np.nanmin(diff), np.nanmax(diff)) if vlim is None else vlim
        im = ax_[0].imshow(diff, vmin=vlim[0], vmax=vlim[1]); plt.colorbar(im, ax=ax_[0])
        ax_[0].set_ylabel("Y"); ax_[0].set_xlabel("X")

        diff = np.reshape(diff, (-1,))
        std_sq2 = np.nanstd(diff)/2**.5
        diff = np.clip(diff, vlim[0], vlim[1])
        ax_[1].hist(diff[np.isfinite(diff)], bins=100, range=vlim); ax_[1].set_xlabel(unit)
        ax_[1].legend(["$\\frac{\sigma}{\sqrt{2}}=%.2f$"%std_sq2])
        
    return axs


def generate_results(rec, predicted=None, mask_xlin=2, SNR_min=30, phaseRMS_max=30, clim=(-0.8,0.8), makepdfs=False, pdfprefix="ku", **report_kwargs):
    """ Generate reports with 'standard_report()' and return the results cross-indexed by all defined tags.
        Result fields 'feedoffsetsHV' & 'errbeamHV' are numpy masked arrays, with masked values represented by NaN's.
        Aperture-plane efficiency & RMS seem to be robust against factors filtered for, therefore not masked.
        
        @param rec: a ResultSet or list of such with data that's already been loaded.
        @param predicted: predicted patterns to de-embed from measured (default None) [ResultSet]
        @param mask_xlin: mask out feedoffsets & errorbeam where beacon yields linear polarisation suppressed by >= this factor (default 2; 3 still leaves suppressed pol with large scatter in offsets)
        @param SNR_min: a cycle for which the co-pol amplitude SNR (mean/stddev) is below this threshold over the cycle, will be masked out (default 30)
        @param phaseRMS_max: a cycle for which the co-pol phase scatter (relative to piecewise linear) is above this threshold is masked as 'True' (default 30).
        @param report_kwargs: optional arguments to override the defaults of 'standard_report()'
        @return: {fid:[HologResults], *(tag:[HologResults])}
    """
    results = {}
    recs = [rec] if isinstance(rec, tuple) else rec # Not straight forward with namedtuples
    for rec in recs:
        r = standard_report(rec, predicted=predicted, clim=clim, makepdf=makepdfs, pdfprefix=pdfprefix, **report_kwargs)
        maskH, maskV = [], [] # (freqs,cycles)
        for i,(pol,bms) in enumerate(zip(rec.beacon_pol, rec.beams)):
            # Initial mask is based on signal-to-noise
            snr, mask = snr_mask(bms, SNR_min, phaseRMS_max)
            for ci in np.flatnonzero(mask): # Cycles to be masked out
                print("INFO: %d @ %.fMHz cycle %d has SNR~%s, masking out feedoffsets & errorbeam"%(rec.fid,rec.f_MHz[i],ci,np.array2string(snr[ci],precision=1).replace("\n",",")))
            
            # Extend mask based on per-pol indicators
            maskH.append(mask); maskV.append(copy.copy(mask))
            # If 'linH' or 'linV' is True then the opposite polarisation will likely have spurious results, esp. feedoffsets!
            linH = (pol not in [None,"RCP","LCP"]) and (abs(pol[0]) > mask_xlin*abs(pol[1]))
            linV = (pol not in [None,"RCP","LCP"]) and (abs(pol[1]) > mask_xlin*abs(pol[0]))
            if linH:
                print("INFO: %d @ %.fMHz has extreme degree of linear polarisation, masking out V-pol feedoffsets & errorbeam"%(rec.fid,rec.f_MHz[i]))
                maskV[-1][:] = True
            elif linV:
                print("INFO: %d @ %.fMHz has extreme degree of linear polarisation, masking out H-pol feedoffsets & errorbeam"%(rec.fid,rec.f_MHz[i]))
                maskH[-1][:] = True
        
        # Convert fields to masked arrays
        maskH, maskV = np.squeeze(maskH), np.squeeze(maskV) # Now matches first one/two dimensions of feedoffsets & errbeam
        maskH3, maskV3 = np.stack([maskH]*3, axis=-1), np.stack([maskV]*3, axis=-1) # Now matches all dimensions of feedoffsets
        maskH4, maskV4 = np.stack([maskH]*4, axis=-1), np.stack([maskV]*4, axis=-1) # Now matches all dimensions of errbeam
        r = r._replace(feedoffsetsH=np.ma.masked_array(r.feedoffsetsH, maskH3, fill_value=np.nan),
                       feedoffsetsV=np.ma.masked_array(r.feedoffsetsV, maskV3, fill_value=np.nan),
                       errbeamH=np.ma.masked_array(r.errbeamH, maskH4, fill_value=np.nan),
                       errbeamV=np.ma.masked_array(r.errbeamV, maskV4, fill_value=np.nan))
        
        results[rec.fid if (rec.cycles is None) else "%d %s"%(rec.fid,rec.cycles)] = [r]
        for t in rec.tags:
            rr = results.get(t,[])
            rr.append(r)
            results[t] = rr
    return results

collect_reports = generate_results # DEPRECATED!

def collate_results(results_a, results_b):
    """ Combines results generated by `report_results()`, by ID's and tags.
        
        @param results_a: results to add, as returned by `report_results()` [dictionary]
        @param results_b: results to add, as returned by `report_results()` [dictionary]
        @return: {fid:[HologResults], *(tag:[HologResults])}
    """
    results = {}
    for a_b in [results_a, results_b]:
        for k,v in a_b.items():
            try:
                results[k].extend(v)
            except:
                results[k] = list(v)
    return results


def meta_report(results, tags="*", tag2label=lambda tag:tag, fspec_MHz=(15000,20000)):
    """ Generate a meta-report of the results from 'generate_results()'.
        
        @param results: {tag:[HologResults]}
        @param tags: a list of specific tags to summarise, in order (default "*" i.e. all tag strings)
        @param fspec_MHz: the frequencies corresponding to the last columns in phase_eff [MHz] """
    if (tags == "*"):
        tags = [t for t in results.keys() if isinstance(t,str)]
    
    A2S = lambda a,fmt="%+.3f": "[%s]"%(" ".join([fmt%f for f in a])) # np.array2string doesn't work correctly with masked arrays!
    
    # Tabulate, and collect result sets for plots
    print("\t\t\tFeed offsets [mm], Reflector phase eff [%] & Error beam [%]:")
    print("Tags      \tEl [deg] [dX_f\tdY_f\tdZ_f]\t\teff@%sGHz\t\tEB [max\t95pct\tstddev\tnoise]"%(A2S(np.asarray(fspec_MHz)/1e3,"%g")))
    
    sets, labels = [], []
    for tag in tags:
        try:
            rr = results[tag]
            sets.append(rr); labels.append(tag)
            # Tabulate key results
            metrics = [[],[]] # H,V
            NFS = len(fspec_MHz) # Number of spec frequencies
            for r in rr:
                for fi,f in enumerate(r.f_MHz):
                    for el,foH,foV,rpH,rpV,ebH,ebV in zip(np.atleast_1d(r.el_deg),np.atleast_2d(r.feedoffsetsH[fi]),np.atleast_2d(r.feedoffsetsV[fi]),
                                                          np.atleast_2d(r.rpeffH[fi]),np.atleast_2d(r.rpeffV[fi]),np.atleast_2d(r.errbeamH[fi]),np.atleast_2d(r.errbeamV[fi])):
                        print("%10s\t%.f       %s\t\t%s\t\t%s"%(tag2label(tag), el, A2S(foH), A2S(rpH[-NFS:],"%.4f"), A2S(ebH*100,"%.2f")))
                        print("          \t         %s\t\t%s\t\t%s"%(A2S(foV), A2S(rpV[-NFS:],"%.4f"), A2S(ebV*100,"%.2f")))
                        metrics[0].append(np.ma.mr_[foH, rpH[-NFS:], ebH*100].filled(np.nan))
                        metrics[1].append(np.ma.mr_[foV, rpV[-NFS:], ebV*100].filled(np.nan))
            if (len(metrics[0]) > 1):
                mH, mV = np.round(np.nanmean(metrics[0],axis=0),3), np.round(np.nanmean(metrics[1],axis=0),3)
                print("       AVG\t         %s\t\t%s\t\t%s"%(A2S(mH[:3]), A2S(mH[3:3+NFS],"%.4f"), A2S(mH[3+NFS:],"%.2f")))
                print("          \t         %s\t\t%s\t\t%s"%(A2S(mV[:3]), A2S(mV[3:3+NFS],"%.4f"), A2S(mV[3+NFS:],"%.2f")))
                mH, mV = np.round(np.nanpercentile(metrics[0],5,axis=0),3), np.round(np.nanpercentile(metrics[1],5,axis=0),3)
                print("      5pct\t         %s\t\t%s\t\t%s"%(A2S(mH[:3]), A2S(mH[3:3+NFS],"%.4f"), A2S(mH[3+NFS:],"%.2f")))
                print("          \t         %s\t\t%s\t\t%s"%(A2S(mV[:3]), A2S(mV[3:3+NFS],"%.4f"), A2S(mV[3+NFS:],"%.2f")))
                mH, mV = np.round(np.nanmedian(metrics[0],axis=0),3), np.round(np.nanmedian(metrics[1],axis=0),3)
                print("       MED\t         %s\t\t%s\t\t%s"%(A2S(mH[:3]), A2S(mH[3:3+NFS],"%.4f"), A2S(mH[3+NFS:],"%.2f")))
                print("          \t         %s\t\t%s\t\t%s"%(A2S(mV[:3]), A2S(mV[3:3+NFS],"%.4f"), A2S(mV[3+NFS:],"%.2f")))
                mH, mV = np.round(np.nanpercentile(metrics[0],95,axis=0),3), np.round(np.nanpercentile(metrics[1],95,axis=0),3)
                print("     95pct\t         %s\t\t%s\t\t%s"%(A2S(mH[:3]), A2S(mH[3:3+NFS],"%.4f"), A2S(mH[3+NFS:],"%.2f")))
                print("          \t         %s\t\t%s\t\t%s"%(A2S(mV[:3]), A2S(mV[3:3+NFS],"%.4f"), A2S(mV[3+NFS:],"%.2f")))
            print("")
        except: # Ignore it if the requested tag isn't in the set
            pass
    
    # Make plots
    plot_offsets_el(sets, labels, figsize=(14,3), fit="lin")
    plot_errbeam_el(sets, labels, figsize=(14,3), extra="RMS")
    for eff_ix in range(-len(fspec_MHz),0):
        plot_eff_el(sets, labels, fspec_MHz=fspec_MHz, eff_ix=eff_ix, figsize=(14,3))
    plot_eff_freq(sets, labels, fspec_MHz=fspec_MHz, figsize=(14,3))


def filter_results(results, exclude_tags=None, f_MHz=None):
    """ Generate a subset of results originally from 'generate_results()'. Omit all HologResults which also appear against 'exclude_tags'
        or not matching 'f_MHz'.
        
        @param results: {tag:[HologResults]}
        @param exclude_tags: a list of tags whose HologResults must be omitted (default None)
        @param f_MHz: (f_min,f_max) frequencies [MHz] to match (default "*" i.e. all)
        @return: {tag:[HologResults]} """
    filtered = {}
    
    # Only include results for which accept_MHs returns True
    accept_MHz = lambda MHz: (f_MHz == "*" ) or (MHz >= np.min(f_MHz) and MHz <= np.max(f_MHz))
    
    exclude_tags = [] if exclude_tags is None else exclude_tags
    omit_tagged = list(iter.chain(*[results.get(xt,None) for xt in exclude_tags])) # HologResults that must be omitted wholesale
    tags = [t for t in results.keys() if (t not in exclude_tags)] # Must filter through these sets
    for tag in tags:
        filtered[tag] = []
        rr = [r for r in results[tag] if (r not in omit_tagged)] # Only continue with HologResults that have not been flagged
        for r in rr: # Select individual measurements in each HologResults based on frequency
            hr = HologResults(el_deg=[],f_MHz=[],feedoffsetsH=[],feedoffsetsV=[],rpeffH=[],rpeffV=[],
                                          rmsH=[],rmsV=[],errbeamH=[],errbeamV=[],info={k:[] for k in r.info.keys()})
            for fi,f in enumerate(r.f_MHz):
                if accept_MHz(f):
                    hr.el_deg.append(r.el_deg[fi])
                    hr.f_MHz.append(r.f_MHz[fi])
                    hr.feedoffsetsH.append(r.feedoffsetsH[fi])
                    hr.feedoffsetsV.append(r.feedoffsetsV[fi])
                    hr.rpeffH.append(r.rpeffH[fi])
                    hr.rpeffV.append(r.rpeffV[fi])
                    hr.rpeffV.append(r.rpeffV[fi])
                    hr.rmsH.append(r.rmsH[fi])
                    hr.rmsV.append(r.rmsV[fi])
                    hr.errbeamH.append(r.errbeamH[fi])
                    hr.errbeamV.append(r.errbeamV[fi])
                    for k in r.keys():
                        hr[k].append(r[k][fi])
            if (len(hr.el_deg) > 0):
                filtered[tag].append(hr)
                
        if (len(filtered[tag]) == 0):
            del filtered[tag]
    
    return filtered
