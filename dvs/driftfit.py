"""
    Functions to fit models to simple "1D scans" (not duplicating scape.beam_baseline, which fits a 2D raster). 
    
    @author: aph@sarao.ac.za
"""
import numpy as np
import scipy.optimize as sop
import scipy.interpolate as interp
from analysis.katsemat import Polynomial2DFit, downsample
import pylab as plt


def _fit_bl_(vis, masks=None, polyorders=[1,1]):
    """ Fit a (low order) polynomial to the first two axes of the data.
        NB: masking (numpy.ma.masked_array) of the input data is ignored for the fit; only the 'vis-baseline' result
        inherits the masking of the input data.
        
        @param vis: real-valued data.
        @param masks: optional (mask0,mask1) to select across first two axes (default None)
        @param polyorders: the orders of the polynomials to fit over the two axes (default [1,1])
        @return: (baseline, vis-baseline) both with the  same shape as 'vis'.
    """
    N_t, N_f, N_p = vis.shape
    f_mesh, t_mesh = np.meshgrid(np.arange(N_f), np.arange(N_t))
    if (masks is None) or ((masks[0] is None) and (masks[-1] is None)):
        masked = lambda x: x
    elif (masks[0] is not None) and (masks[1] is not None):
        masked = lambda x: x[np.ix_(*masks)]
    elif (masks[0] is not None):
        masked = lambda x: x[masks[0],...]
    elif (masks[1] is not None):
        masked = lambda x: x[:,masks[1],...]
    x_p, y_p = masked(t_mesh), masked(f_mesh)
    
    # The fit seems to be much improved if we remove bias in both the data, as well as in all axes with gaps in it
    x0, y0, v0 = np.mean(x_p), np.mean(y_p), np.nanmean(vis, axis=0)
    z_p = masked(vis - v0)
    model = Polynomial2DFit(polyorders)
    bl = [model.fit([x_p-x0, y_p-y0], z_p[...,p])([t_mesh-x0, f_mesh-y0]) for p in range(N_p)]
    bl = np.stack(bl, axis=-1) # t,f,p
    bl += v0 # Restore bias
    
    v_nb = vis - bl # The data with the fitted baseline subtracted
    return bl, v_nb


def _fit_bm_(vis, t_axis, force=False, sigmu0=None, debug=True):
    """ Fits a Gaussian, plus a first order polynomial for the baseline along the first axis of the data (which fails to converge
        if the baseline slope exceeds the beam height). This fit is repeated sequentially and independently along the second axis,
        consequently execution time scales linearly with the size of the second axis.
        
        @param vis: real-valued data arranged as (time, freq, prod)
        @param t_axis: the intervals along the first axis, in terms of which sigma & mu will be defined.
        @param force: True to return starting estimate (rather than NaN's) if solution doesn't converge (default False).
        @param sigmu0: first estimate for [sigma,mu], or None to use default starting estimate (default None)
        @return: [baseline, beam] each shaped like vis; [sigmaH,sigmaV] each shaped like vis axis 1; [muH,muV] like vis axis 1
    """
    # TODO: add a smoothness constraint for mu over the first two axes (time & frequency)
    G = lambda ampl,sigma,mu: abs(ampl)*np.exp(-1/2.*(t_axis-mu)**2/sigma**2) # 1/sigma/sqrt(2pi)*... is absorbed in amplitude term
    B = lambda oH,oV,sH,sV, aH,aV,sigmaH,sigmaV,muH,muV: np.c_[oH+sH*t_axis + G(aH,sigmaH,muH),
                                                               oV+sV*t_axis + G(aV,sigmaV,muV)]
    W = lambda oH,oV,sH,sV, aH,aV,sigmaH,sigmaV,muH,muV: np.c_[0.4 + G(1,(sigmaH+sigmaV),(muH+muV)/2), # Weights to emphasize the beam peak over the baseline
                                                               0.4 + G(1,(sigmaH+sigmaV),(muH+muV)/2)]
    
    N_t, N_f, N_p = np.shape(vis)
    assert (N_p==2), "_fit_bm_() is hard-coded for data shaped like (*,*,2), not %s"%np.shape(vis)
    
    # Starting estimates: ampl, sigma, mu
    if sigmu0 is None:
        mu0 = np.median(t_axis[np.ma.any(vis>np.nanpercentile(vis,80,axis=(0,1)), axis=(1,2))])
        sigma0 = N_t/9. # Reasonable guess to start for typical scans -- no fit expected if 4*sigma > N_t
    else:
        sigma0, mu0 = sigmu0
    if np.isnan(mu0): # This happens in some pathological cases (e.g. channel 0), return NaN's
        return (np.nan+vis), (np.nan+vis), [[np.nan]*2]*N_f, [[np.nan]*2]*N_f
    
    A0 = np.nanpercentile(vis[int(mu0-10):int(mu0+10),...].data,95,axis=0) - np.nanpercentile(vis.data,5,axis=0) # (freq,prod) Use .data since can't use ma.compressed() - it discards dimensions
    
    bl, bm, sigma, mu = [], [], [], [] # Arranged as (freq,time,prod) and (freq,prod)
    vis = vis / A0 # Normalize amplitudes, so that both pols contribute similarly to optimization metric
    for f in range(N_f): # Fit per frequency channel
        if debug and (f%10 == 0):
            print("INFO: Fitting channel %d of %d"%(f,N_f))
        v_nb = vis[:,f,:]
        p0 = [0,0, 0,0, 1,1, sigma0,sigma0, mu0,mu0]
        p, s, _,_, _, w = sop.fmin_powell(lambda p: np.nansum(W(*p)*(B(*p)-v_nb)**2), p0, full_output=True, disp=False)
        ss = np.nansum((B(*p)-v_nb)**2, axis=(0,1))
        
        # Repeat the fit if necessary
        if ((w == 1) or (np.min(p[-2:]) < 0 or np.max(p[-2:]) >= N_t)): # warning OR bore sight transit out-of-bounds = likely invalid fit
            if debug:
                print("INFO: beam fitting failed to converge on first attempt, re-trying with better starting estimates.")
            # Update p0 based on the best of the two pols and retry
            bb = np.argmin(ss) # Best of the two pols
            if ((p[-2+bb] < 0) or (p[-2+bb] >= N_t)): # In some pathological cases the lowest residual has mu out of bounds
                bb = abs(1-bb) # 0->1, 1->0
            p0 = p0[:-4] + [p[bb+len(p0[:-4])+i] for i in [0,0,2,2]] # sigma & mu from the best of the fitted pols
            p, s, _,_, _, w = sop.fmin_powell(lambda p: np.nansum(W(*p)*(B(*p)-v_nb)**2), p0, full_output=True, disp=False)
            ss = np.nansum((B(*p)-v_nb)**2, axis=(0,1))
            
            if ((w == 1) or (np.min(p[-2:]) < 0 or np.max(p[-2:]) >= N_t)): # Unrecoverable after two attempts
                if force:
                    print("WARNING: beam fitting failed to converge (SS: %g~%s), using 2nd order estimate in stead" % (s, str(ss/np.max(ss))))
                    p = p0
                else:
                    if debug:
                        print("INFO: beam fitting failed to converge (SS: %g~%s), channel %d gets NaNs instead of %s" % (s, str(ss/np.max(ss)), f, str(p)))
                    bl.append(np.nan+v_nb); bm.append(np.nan+v_nb); sigma.append([np.nan]*2); mu.append([np.nan]*2)
                    continue
            elif debug:
                print("INFO: beam fitting successful on second attempt")
        
        p_bl, p_bm = p[:-6], p[-6:]
        bl.append(B(*(list(p_bl)+[0,0,1,1,0,0]))); bm.append(B(*(list(0*p_bl)+list(p_bm))))
        sigma.append(p[-4:-2]); mu.append(p[-2:])
    bl, bm = np.stack(bl,axis=1)*A0, np.stack(bm,axis=1)*A0 # Change from (freq,time,prod) to (time,freq,prod) and reverse the earlier normalization
    sigma, mu = np.asarray(sigma), np.asarray(mu)
    
    return bl, bm, sigma, mu


def _mask_jumps_(data, jump_zone=0, fill_value=np.nan, thresh=10, debug=False):
    """ Automatically masks discontinuities in first two axes, independently across fourther axes. Builds on top
        of any existing mask, if present.
        @param data: N dimensional (N>2), real data (must be a masked array).
        @param jump_zone: a single value for both axes, or a tuple; >=0 to blank out this many samples either side of a jump,
                          <0 for no blanking (default 0).
        @param fill_value: fill value for the masked data in the returned array (default numpy.nan)
        @param thresh: threshold for flags, as a multiple of median(diff(2)) (default 10)
        @return: (masked_data, mask0, mask1). Masks are N-1 dimensional with True where data is masked out. """
    N_t, N_f = data.shape[:2]
    t_axis = np.arange(N_t)
    f_axis = np.arange(N_f)
    jump_zone = [jump_zone]*2 if np.isscalar(jump_zone) else jump_zone
    
    fmask = np.full([N_f]+list(data.shape[2:]), True)
    if (jump_zone[1] > 0): # Flag out frequency axis with extreme jumps - like RFI
        flags = np.ma.median(data, axis=0) # Power over frequency, product
        flags = np.abs(np.diff(flags/np.nanmedian(flags, axis=0), 2, axis=0)) # Discontinuity in slope identify edges
        flags[np.isnan(flags)] = np.nanmax(flags) # Avoid warning message in 'flags>' below
        # Remember we flag out 2*jump_zone samples around each jump, so be careful to not ID too many jumps
        jump_thresh = thresh*np.nanmedian(flags[flags>np.nanmedian(flags[jump_zone[1]:-1-jump_zone[1]])])
        if debug: # Plot the second order derivatives, which show the jumps
            plt.figure(figsize=(12,4))
            plt.subplot(1,2,2); plt.title("axis 1")
            plt.plot(f_axis[2:], flags)
            plt.plot(f_axis[2:], jump_thresh+0*flags); plt.ylim(0,10*jump_thresh)
        fmask = np.r_[flags[0:2],flags] > jump_thresh # Expand to match shape of data again & apply threshold
        for k in range(jump_zone[1]//2,0,-1): # Block out 'jump_zone' samples either side of a jump
            fmask = fmask | np.roll(fmask,k, axis=0) | np.roll(fmask,-k,axis=0)
        data = np.ma.masked_array(data, np.stack([fmask]*N_t,axis=0), fill_value=fill_value) # Same mask for all time samples
    fmask = np.count_nonzero(data.mask, axis=0) > 0.1*N_t # Collapse time axis @10% & combine with pre-existing mask, if any.
    #fmask = np.any(data.mask, axis=0) # There's a chicken-and-egg situation, which we solve by using the preceding line instead of this one

    tmask = np.full([N_t]+list(data.shape[2:]), True)
    if (jump_zone[0] > 0): # Flag out time axis with extreme jumps - like RFI, ND at start & end, or gain jumps?
        flags = np.ma.median(data, axis=1) # Power over time, product
        flags = np.abs(np.diff(flags/np.max(flags, axis=0), 2, axis=0)) # Discontinuity in slope identify edges which is not expected for "good drift scans"
        flags[np.isnan(flags)] = np.nanmax(flags) # Avoid warning message in 'flags>' below
        # Remember we flag out 2*jump_zone samples around each jump, so be careful to not ID too many jumps
        jump_thresh = thresh*np.nanmedian(flags[flags>np.nanmedian(flags[jump_zone[0]:-1-jump_zone[0]])])
        if debug: # Plot the second order derivatives, which show the jumps
            plt.subplot(1,2,1); plt.title("axis 0")
            plt.plot(t_axis[2:], flags)
            plt.plot(t_axis[2:], jump_thresh+0*flags); plt.ylim(0,10*jump_thresh)
        tmask = np.r_[flags[0:2],flags] > jump_thresh # Expand to match shape of data again & apply threshold
        for k in range(jump_zone[0]//2,0,-1): # Block out 'jump_zone' samples either side of a jump
            tmask = tmask | np.roll(tmask,k, axis=0) | np.roll(tmask,-k,axis=0)
        data = np.ma.masked_array(data, np.stack([tmask]*N_f,axis=1), fill_value=fill_value) # Same mask for all frequency bins
    tmask = np.any(data.mask[:,~np.any(fmask,1),:], axis=1) # Collapse freq axis & combine with pre-existing mask, if any.
    
    if debug and (len(data.shape) == 3):
        ax = plt.subplots(1,2, figsize=(12,6))[1]
        ax[0].imshow(data[...,0], aspect='auto')
        ax[1].imshow(data[...,1], aspect='auto')
        ax[0].set_ylabel("axis0"); ax[0].set_xlabel("axis1"); ax[1].set_xlabel("axis1")
    
    return data, tmask, fmask


def fit_bm(vis, n_chunks=0, freqchans=None, timemask=None, jump_zone=0, debug=0, debug_label=None):
    """ Fit a Gaussian bump plus a baseline defined by a 2nd order polynomial to the time (spatial) axis and
        a first order polynomial over frequency.
        Note: execution time scales linearly with n_chunks, typically at 0.3sec/channel.

        @param vis: data arranged as (time,freq,products)
        @param n_chunks: > 0 to average the frequency range into this many chunks to fit beams, or <=0 to fit band average only (default 0).
        @param freqchans: selector to filter the indices of frequency channels to use exclusively to identify jumps (default None).
        @param timemask: selector to filter out samples in time (default None)
        @param jump_zone: >=0 to blank out this many samples either side of a jump, <0 for no blanking (default 0).
        @param debug: 0/False for no debugging, 1/True for showing the fitted 'mu & sigma', 2 to also show the raw & model data (default 0)
        @param debug_label: text to label debug figures with (default None)
        @return: baseline, beam (Power, same shapes as vis), sigma (Note 1,3), mu (Note 2,3).
                 Note 1: sigma is the standard deviation of duration of main beam transit, per freq, so HPBW = sqrt(8*ln(2))*sigma [in units of time dumps]
                 Note 2: mu is times of bore sight transit per frequency [in units of time dumps]
                 Note 3: sigma & mu are masked arrays, with non-finite and outlier values masked.
    """
    # The fitting easily fails to converge if there's too large a slope, so the approach is
    # 1. fit a rough provisional baseline on a masked sub-set (fundamental limitation in how good we can mask e.g. over frequency) 
    # 2. fit a beam + delta baseline on the residual (data-baseline), each frequency channel independently.
    # 3. combine the provisional and delta baselines to form the final baseline
    
    N_t, N_f = vis.shape[:2]
    t_axis = np.arange(N_t)
    f_axis = np.arange(N_f)
    
    mask = np.isnan(vis) # False to keep data
    if freqchans is not None:
        m2 = np.full(vis.shape, True); m2[:,freqchans,:] = False
        mask |= m2 # Discard those that remain True
    if timemask is not None:
        m2 = np.full(vis.shape, True); m2[timemask,...] = False
        mask |= m2 # Discard those that remain True
    vis = np.ma.masked_array(vis, mask, fill_value=np.nan) # No need for copy=True because vis never gets modified
    vis, tmask, fmask = _mask_jumps_(vis, jump_zone=jump_zone, fill_value=np.nan) # Using nan together with np.nan* below
    tmask = ~np.any(tmask, axis=1) # True to keep data; collapsed across products, since code below doesn't yet cope with mask per pol.
    fmask = ~np.any(fmask, axis=1) # Includes freqchans
    # 0
    if (debug >= 2):
        fig, axs = plt.subplots(2,1, figsize=(12,6))
        fig.suptitle(debug_label)
        axs[0].set_title("Provisional baseline fitting")
        axs[0].plot(t_axis, np.nanmean(vis[:,fmask,:], axis=1))
        axs[0].plot(t_axis[tmask], np.nanmean(vis[tmask,:,:], axis=1))
        axs[0].set_ylabel("Power [linear]"); axs[0].grid(True)
        # axs[1] is completed in steps up to 2 below
    
    # 1. Fit & subtract provisional baseline through first x% and last x% of the time series.
    # Ideally this should vary over frequency, but _fit_bl_ needs regular shaped, un-masked data, and it might not be worthwhile to transform vis to (time, angle/HPBW, prod)
    _tmask = np.array(tmask, copy=True); _tmask[N_t//4:-N_t//4] = False
    bl, vis_nb = _fit_bl_(vis, (_tmask,fmask), polyorders=[1,2])
    if (debug >= 2):
        axs[0].plot(t_axis, np.mean(bl[:,fmask,:], axis=1))
    
    # 2. Fit beam+delta baseline on the integrated (band average) & force a non-NaN solution.
    vis0_nb = np.ma.mean(vis_nb[:,fmask,:], axis=1) # Integrated power in H & V, over time
    dbl, bm, sigma, mu = _fit_bm_(np.moveaxis([vis0_nb],0,1), t_axis, force=True, debug=False) # passing in (time,freq,prod)
    dbl, bm, sigma, mu = np.repeat(dbl,N_f,axis=1), np.repeat(bm,N_f,axis=1), np.repeat(sigma,N_f,axis=0), np.repeat(mu,N_f,axis=0) # Repeat along (existing) freq axis
    
    if (n_chunks <= 0): # Asked for the band average fits are copied across frequency
        if (debug >= 2):
            axs[1].set_title("Baselines subtracted")
            axs[1].plot(t_axis, np.nanmean((vis-bl-dbl)[:,fmask,:],axis=1), label="Baselines subtracted")
            axs[1].plot(t_axis, np.nanmean(bm[:,fmask,:],axis=1), label="Fitted models")
            axs[1].set_xlabel("Time [samples]"); axs[1].set_ylabel("Power [linear]"); axs[1].legend(); axs[1].grid(True)

    else: # 2. Fit beam+delta baseline on a per-frequency bin basis, using band average as starting estimate
        ch_res = int(len(f_axis[fmask])/n_chunks)
        sigmu0 = [np.nanmean(sigma), np.nanmean(mu)]
        # Reduce resolution before fitting (to speed up)
        chans = np.asarray(downsample(f_axis[fmask], ch_res, method=np.nanmean, trunc=True), int)
        vis_nb = downsample(vis_nb[:,fmask,:], ch_res, axis=1, method=np.nanmean, trunc=True)
        # Fit each remaining channel independently, with NaN's where fit doesn't converge
        dbl, bm, sigma, mu = _fit_bm_(vis_nb, t_axis, force=False, sigmu0=sigmu0, debug=debug>0)
        # Interpolate to restore original frequency resolution
        dbl = interp.interp1d(chans, dbl, 'quadratic', axis=1, bounds_error=False)(f_axis)
        bm = interp.interp1d(chans, bm, 'quadratic', axis=1, bounds_error=False)(f_axis)
        mu = interp.interp1d(chans, mu, 'quadratic', axis=0, bounds_error=False)(f_axis)
        sigma = interp.interp1d(chans, sigma, 'quadratic', axis=0, bounds_error=False)(f_axis)
        # Mask out all suspicious results
        _sigma = np.nanmedian(sigma)
        mask = ~np.isfinite(sigma) # False to keep data
        mask[~mask] |= (np.abs(sigma[~mask]) > 2*_sigma)  | (np.abs(sigma[~mask]) < 0.5*_sigma) # Split it like this to avoid unnecessary RuntimeWarnings where sigma is nan!
        sigma = np.ma.masked_array(sigma, mask, fill_value=np.nan)
        mu = np.ma.masked_array(mu, mask, fill_value=np.nan)
        if (debug >= 2):
            axs[1].set_title("Baselines subtracted")
            _a, _b = (vis-bl-dbl)[:,fmask,:], bm[:,fmask,:]
            for i in range(2): # Two halves of the band, separately
                _f = slice(int(i*_a.shape[1]/2), int((i+1)*_a.shape[1]/2))
                axs[1].plot(t_axis, np.nanmean(_a[:,_f,:],axis=1), '-', label="Measured %d/2"%(i+1))
                axs[1].plot(t_axis, np.nanmean(_b[:,_f,:],axis=1), '--', label="Fitted %d/2"%(i+1))
            axs[1].set_xlabel("Time [samples]"); axs[1].set_ylabel("Power [linear]"); axs[1].legend(); axs[1].grid(True)
            
            fig, axs = plt.subplots(1,2, figsize=(12,6))
            fig.suptitle(debug_label)
            resid = ((vis-bm-bl-dbl)/np.max(bm,axis=0) * 100) # Percentage, masked
            for p in [0,1]:
                axs[p].set_title("Model residuals P%d [%%]"%p)
                im = axs[p].imshow(resid[:,:,p], origin="lower", aspect='auto', vmin=-10,vmax=10, cmap=plt.get_cmap('viridis'))
                axs[p].set_xlabel("Frequency [channel]")
            axs[0].set_ylabel("Time [samples]"); plt.colorbar(im, ax=axs[1]) 
            
    # 3. Update the provisional baseline so that bl+mb ~ vis
    bl += dbl
    
    if (debug >= 1):
        fig, axs = plt.subplots(2, 1, figsize=(12,6))
        fig.suptitle(debug_label)
        axs[0].plot(f_axis, mu); axs[0].set_ylabel("Peak response time 'mu'\n[time samples]"); axs[0].grid(True)
        axs[1].plot(f_axis, sigma); axs[1].set_ylabel("HP crossing duration 'sigma'\n[time samples]"); axs[1].grid(True)
        axs[1].set_xlabel("Frequency [channel]")
    
    return bl, bm, sigma, mu
