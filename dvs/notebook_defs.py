"""
    Functions for standard DVS notebooks, for which the tight integration with the notebooks makes it infeasible to
    promote to more general library functions.
    
    
    @author aph@sarao.ac.za
"""
import pylab as plt
import numpy as np
from . import util, tracking
import sys, os
sys.path.append(os.path.dirname(__file__)+'/../bin')
import baseline_cal_reduction


def fit_delaymodels(cbids, ref_ant, args, pols="H,V", make_plots=True, rfi_mask="../catalogues/rfi_mask.txt", **kwargs):
    """@param cbids: datasets comprised of 'track' on targets, all must have exactly the same telescope configuration!
                   NB: Can only combine datasets if center frequency, number of channels and dump rate are the same.
                       ALSO only if delay model is the same (no error message - just silently skips all but the first one).
       @param args: text, typically "-f 0,4096 -s 0.9" may also add e.g. " --discard-tracklength 20" to drop tracks with unidentified hang/crash
       @param kwargs: alternative to args, passed to baseline_cal_reduction.reduce_delays()
       @return: {pol:([ants, enu, sigma_enu, old_enu, cable_len, sigma_c_l, old_c_l, niao, sigma_niao, old_niao])} for all polarisations
    """
    cbids = np.atleast_1d(cbids)
    
    # CAUTION: Very primitive conversion!
    args = args.replace("--pre-average", "--pre-average True")
    args = [_ for _ in args.split(" ") if len(_)>0] # Omit blanks
    _kwargs = {k.replace("--","-")[1:]:v for k,v in zip(args[0::2],args[1::2])} # Drop first - & -- from keys
    _kwargs = {k.replace("-","_"):v for k,v in _kwargs.items()}
    for k,v in _kwargs.items(): # Type conversion
        try:
            _kwargs[k] = float(v)
        except:
            pass
    _kwargs.update(kwargs)
    _kwargs["discard_tracklength"] = _kwargs.get("discard_tracklength", 20) # Longer than 20sec is probably a controller that hangs
    
    print(">>", cbids, _kwargs)
    
    ants = _kwargs.pop('ants',None)
    freq_chans = _kwargs.pop('f',None)
    discard_sigma = _kwargs.pop('s')
    
    # cbid2fn = util.cbid2url
    for cbid in cbids: # Ensure datasets are cached
        util.open_dataset(cbid, cache_root="./l1_data")
    cbid2fn = lambda cbid: f"./l1_data/{cbid}/{cbid}_sdp_l0.full.rdb"
    
    results = {}
    for pol in pols.split(","):
        results[pol] = baseline_cal_reduction.reduce_delays([cbid2fn(cbid) for cbid in cbids], ants, freq_chans, rfi_mask, 
                                                            discard_sigma, pol, ref_ant, make_plots=make_plots, **_kwargs)
        plt.show()
    
    return results


def plot_delay_convergence(rr, ant, update_ref=None):
    """ @param rr: list of {'H':(antnames, ...),'V':(antnames, ...)} as returned by fit_delaymodels()
        @param ant: name of antenna for which to plot convergence results
        @param update_ref: {refant:refpos} e.g. {'e107':[e,n,u,H,V,NIAO]} (default None)
        @return: mean position model for ant """
    pols = sorted(rr[0].keys())
    r_by_pol = [[] for p in pols] # Results only for this ant, excluding antnames
    for r in rr:
        if (ant in r[pols[0]][0]):
            ant_idx = r[pols[0]][0].index(ant)
            for i,p in enumerate(pols):
                r_by_pol[i].append([_[ant_idx] for _ in r[p][1:]])
        else:
            for i,p in enumerate(pols):
                r_by_pol[i].append([[np.nan]*3, [np.nan]*3, [np.nan]*3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    
    if (update_ref is not None): # Attempt to update positions
        for refant, refpos in update_ref.items(): # TODO handle multiple
            ref_r = [np.r_[refpos[0:3]], 0, np.r_[refpos[0:3]], refpos[3], 0, refpos[4], refpos[5], 0, refpos[5]]
            for i,r in enumerate(rr):
                refant_idx = r[pols[0]][0].index(refant)
                for p, r_to_update in zip(pols, r_by_pol):
                    delta = [ref-_[refant_idx] for ref,_ in zip(ref_r,r[p][1:])]
                    r_to_update[i] = [_+d for d,_ in zip(delta,r_to_update[i])]

    axs = plt.subplots(4,2, width_ratios=[3,1], sharex='col', figsize=(12,8))[1]
    plt.suptitle(ant)
    # Series of ENUC along with 3*sigma error bars
    for rrp,fmt in zip(r_by_pol, "_./\\"):
        for i,r in enumerate(rrp):
            for j,p in enumerate("ENU"):
                axs[j][0].errorbar(i, r[0][j], 3*r[1][j], fmt=fmt, capsize=3)
            axs[3][0].errorbar(i, r[3], 3*r[4], fmt=fmt, capsize=3)
        for j,p in enumerate("ENU"):
            axs[j][0].hlines(r[2][j], 0,len(rrp)-1, 'k'); axs[j][0].set_ylabel(p)
        axs[3][0].hlines(r[5], 0,len(rrp)-1, 'k'); axs[3][0].set_ylabel("C")
        for j in range(4):
            axs[j][0].grid(True)

    # Histograms of ENUC
    mean_pos = []
    for j,p in enumerate("ENU"):
        values = np.r_[[r[0][j] for r in (r_by_pol[0]+r_by_pol[-1])]] # All pols together
        sigmas = np.r_[[1/r[1][j] for r in (r_by_pol[0]+r_by_pol[-1])]]
        mask = np.isfinite(values) & np.isfinite(sigmas)
        try:
            mean_pos.append(np.average(values[mask], weights=sigmas[mask]))
        except:
            mean_pos.append(np.mean(values[mask]))
        axs[j][1].hist(values[mask], orientation='horizontal'); axs[j][1].legend(["~%.2f"%mean_pos[-1]])
    for p,rrp in zip(pols,r_by_pol): # This is unique by pol
        values = np.r_[[r[3] for r in rrp]]
        sigmas = np.r_[[1/r[4] for r in rrp]]
        mask = np.isfinite(values) & np.isfinite(sigmas)
        try:
            mean_pos.append(np.average(values[mask], weights=sigmas[mask]))
        except:
            mean_pos.append(np.mean(values[mask]))
        axs[3][1].hist(values[mask], orientation='horizontal', label="%s ~ %.2f"%(p,mean_pos[-1]))
    axs[3][1].legend()
    # Also add NIAO
    values = [r[-3] for r in (r_by_pol[0]+r_by_pol[-1])] # All pols together
    mean_pos.append(np.nanmean(values))

    print(ant, "-->", ' '.join(['%.2f'%_ for _ in mean_pos]))
    return mean_pos


def fit_centroids(ant, cbids, mask_S4=False, kind='cardioid', scans="track", strict=True, mkat_array=False,
                  rfi_mask_fn="../catalogues/rfi_mask.txt", root='./l2_data', debug=False, **kwargs):
    """ Fit centroids to pointing datasets with the `dvs.tracking` module.
    
        @param ant: a specific antenna ID, or 'scan_ants' to process all.
        @param root: path string or function like root(ant) => f'../{ant}/tmp'
        @param mkat_array: True if data was recorded in MeerKAT instead of DVS array.
        @param mask_S4: True to omit 3.0-3.4GHz (default False). """
    if mkat_array:
        for k in util.ksl.SENSOR_PORTALS.keys(): # Undo the block that's currently activated in 'util'
            if (".mkat." in k): util.ksl.SENSOR_PORTALS[k] = 1
    phased_up = kwargs.pop('phased_up', None)
    chans = kwargs.pop('chans', None)
    for cbid in np.atleast_1d(cbids):
        ants = [ant] if (ant != 'scan_ants') else util.open_dataset(cbid).obs_params['scan_ants'].split(',')
        for _ant in ants:
            print(_ant, cbid)
            _root = root(_ant) if callable(root) else root
            cacheds = ds = util.open_dataset(cbid, 'm' if mkat_array else _ant, cache_root="./l1_data")
            _chans = chans
            if (not callable(_chans)) and ('freq_MHz' not in kwargs.keys()):
                rfi_mask = (ds.channels<10) | (len(ds.channels)-10 < ds.channels)
                if mask_S4: # S-band beam features
                    rfi_mask |= (3000e6 < ds.freqs) & (ds.freqs <=3500e6)
                if (rfi_mask_fn):
                    rfi_mask |= util.load_rfi_static_mask(rfi_mask_fn, ds.freqs)
                _chans = ~rfi_mask if (_chans is None) else (_chans & ~rfi_mask)
            track_ants = ds.obs_params['track_ants'].split(',') if ('track_ants' in ds.obs_params.keys()) else [None]
            for track_ant in track_ants:
                try:
                    output_filepattern = f"%s_%s_{track_ant}_circular_pointing.csv" if (track_ant is not None) else f"%s_%s_point_source_scans.csv"
                    output_filepattern = None if (_root is None) else _root+'/'+output_filepattern
                    tracking.reduce_pointing_scans(ds, _ant, track_ant=track_ant, kind=kind,scans=scans, chans=_chans, strict=strict,
                                                   phased_up=(phased_up==True) or ((phased_up==None) and np.max(ds.freqs)<4e9), # Typically no advantage for Ku-band
                                                   output_filepattern=output_filepattern, debug=debug, verbose=True, **kwargs)
                except Exception as e:
                    import traceback
                    print("WARNING: Failed to fit centroids!", e, traceback.format_exc())
        cacheds.del_cache()
