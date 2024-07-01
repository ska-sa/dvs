#!/usr/bin/python
"""
    Script that analyses fast gain stability measurement at SCP
    Typical use:
        analyze_fastgain.py --ant m021 /var/kat/archive4/data/RTS/telescope_products/2018/06/04/1528106528.h5
    
    @author aph@sarao.ac.za
"""

import numpy as np
import pickle
from .util import open_dataset
from pylab import figure, subplots, plot, psd, imshow, colorbar, legend, xlabel, ylabel, subplot, ylim, title, suptitle
import pylab as plt


def sliding_rms(x, win_length):
    """ @return: RMS computed over sliding windows on x, zero-padded to align with x. """
    W = np.min([len(x), int(win_length+0.1)])
    rms1 = lambda x_block: (np.sum((x_block-np.average(x_block))**2)/float(W))**.5
#     return katsemat.sliding_window(x, x, win_length, func=rms1, overlap=1) # TODO: use this in future
    results = list(0*x) # Allocate & fill with zeros
    for i in range(len(x)-W): # Process sliding blocks of length N
        results[W//2+i] = rms1(x[i:i+W])
    return np.asarray(results)

def fit_avg(x, win_length):
    """ @return: block average over x."""
    W = np.min([len(x), int(win_length+0.1)])
    results = list(x) # Copy - this ensures padding with original values for last ~window
    for i in range(0,len(x)-W+1,W-1): # W-1 steps means the last preceding point is the first next point 
        results[i:i+W] = np.ones(W)*np.average(x[i:i+W])
    return np.asarray(results)

def calculate_allanvariance(data):
    """
        Calculates the Allan variance series, as proposed by Rau, Schneider & Vowinkel
        - see "Characterization and Measurement of Radiometer Stability"
        @param data: measurements spaced at regular intervals
        @return: (Ks, sA2) - the number of integration intervals and the Allan variance^2,
        respectively.
    """ 
    # Functions to calculate the Allan variance:
    # (The python compiler does a good job at optimizing this, so I focused on readability.)
    def R(i, K, x):
        y = [x[i*K + n] for n in range(K)]
        return np.mean(y)
    
    def s2(K, x):
        M = int(len(x)/K)-1
        y = [(R(i+1, K, x) - R(i, K, x))**2 for i in range(M)]
        return np.mean(y)/2.
    
    # Calculate the Allan variances vs. integration intervals:
    Ks = range(1,int(len(data)/3))
    sA2 = np.asarray([s2(K, data) for K in Ks])
    return Ks, sA2

def plot_allanvar(x, dt=1, time=None, sfact=1., xylabels=("time [sec]","amplitude [lin]"), label=None, title="", grid=False, plot_raw=False, normalize=True, figs=None):
    """
        Creates the Allan variance plot for the (regularly sampled) data, as
        proposed by Rau, Schneider & Vowinkel
        - see "Characterization and Measurement of Radiometer Stability"
        @parm x: time series data - in a linear scale!
        @param dt: the time increment between successive data measurements.
        @param time: the regularly spaced time vector, if available.
        @param sfact: scale factor to apply to deviations e.g. in case of samples being differences (default 1)
        @param xylabels: pair of (xlabel, ylabel) for plots (default "time [sec]", "amplitude [lin]").
        @param label: the legend label for this data series (default None).
        @param title: a title to use for the figure instead of the automatic one (default None)
        @param grid: True to show the grid on all of the plots (default False)
        @param plot_raw: True to plot the time series and PSD thereof (default False)
        @param normalize: True to normalize the data to the average value before analysis (default True)
        @param figs: a list of length 2 (or 1 if not plot_raw) containing existing pylab
        figures to plot over, or None to create new ones (default None).
        @return: the list of figures created / re-used ("raw" is at index 0, "avar" is at -1)
    """
    time = dt*np.arange(len(x)) if (time is None) else time
    time = time - time[0] # Ensure there are no confusing sample offsets
    p_data = np.asarray(x)
    if normalize: # Legacy support: has no impact on the slope of Allan variance curve
        p_data = p_data/np.mean(p_data)
    
    figs = [None]*2 if (figs is None) else figs
    
    if (plot_raw): # Plot the raw data
        if figs[0]:
            plt.sca(figs[0].axes[-1])
        else:
            figs[0] = figure(figsize=(14,6))
        plot(time, p_data, ".", label=label); legend()
        xlabel(xylabels[0]); ylabel(xylabels[1]); plt.grid(grid)
        plt.title("Time series: %s"%title)
    
    K, sA2 = calculate_allanvariance(p_data)
    sA2 *= sfact**2 # sfact applies to stddev
    # Calculate and plot the Allan variances vs. integration time:
    if (dt > 0.1): ndecimals = 1
    elif (dt > 0.01): ndecimals = 2
    elif (dt > 0.001): ndecimals = 3
    else: ndecimals = 4
    if figs[-1]:
        plt.sca(figs[-1].axes[-1])
    else:
        figs[-1] = figure(figsize=(14,6))
    plot(np.log10(K), 0.5*np.log10(sA2), ".--", label=label); legend()
    plot(np.log10(K), 0.5*np.log10(sA2[0]/K), "-") # White noise, for comparison
    ylabel(r"$\log_{10}(\sigma_A)$"); xlabel(xylabels[0]); plt.grid(grid)
    plt.title("Allan Variance: %s"%title)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda k, pos: round(dt*10**k,ndecimals)))
    
    return figs


def analyse(h5, ant, t_spike_start, t_spike_end, channels=None, vs_freq=False, T_interval=5, sigma_spec=0.10, cycle_50pct=False,
           xK=[1.03,1.05,1.1,1.15]):
    """
        @param t_spike_start, t_spike_end: start & end times of noisy time series to be excluded from analysis, in [sec]
        @param channels: channel indices (list or slice) to select (default None)
        @param vs_freq: True to make plots vs. frequency rather than channel number (default False)
        @param T_interval: interval for sliding window to evaluate over, in seconds (default 5)
        @param sigma_spec: the reference limit in percent to indicate on the final figures (default 0.10)
        @param cycle_50pct: True to process the data as the difference between consecutive samples (default False)
        @param xK: acceptable scale factors for stability threshold (default [1.03,1.05,1.1,1.15])
    """
    filename = h5.name.split()[0].split(" ")[0] # "rdb files are different
    # Select the correct scan. It's always the last 'track'
    h5.select(reset="TFB", scans='track')
    h5.select(scans=h5.scan_indices[-1], ants=[ant])
    if (channels is not None):
        h5.select(channels=channels)
    fft_shift, gains = get_fft_shift_and_gains(h5)
    
    t = h5.timestamps-h5.timestamps[0]
    dt = h5.dump_period
    if (abs(dt/(t[1]-t[0])-1) > 0.1):
        print("Discrepancy between dump period and recorded time intervals, using latter")
        dt = np.mean(np.diff(h5.timestamps))
    pols = [i for i,x in enumerate(h5.corr_products) if x[0][-1]==x[1][-1]] # Only XX & YY
    p_h = np.abs(h5.vis[:,:,pols[0]].squeeze())
    p_v = np.abs(h5.vis[:,:,pols[1]].squeeze())
    p_hv = np.abs(h5.vis[:,:,[i for i,x in enumerate(h5.corr_products) if x[0][-1]!=x[1][-1]][0]].squeeze())
    if cycle_50pct: # Take differences of consecutive [ON,OFF,ON,OFF...] samples -> [ON-OFF,OFF-ON,ON-OFF...],
                    # convert all to |ON-OFF| and finally trim time vector to match
        p_h = np.diff(p_h, axis=0); p_h[::2,:] *= -1; p_h = np.abs(p_h)
        p_v = np.diff(p_v, axis=0); p_v[::2,:] *= -1; p_v = np.abs(p_v)
        p_hv = np.diff(p_hv, axis=0); p_hv[::2,:] *= -1
        t = t[1:]
        # Take only every second difference, since subsequent ones are not 100% independent
        dt *= 2.
        t = t[:-1:2]           # Taking the averages below messes slightly with AVAR slopes
        p_h = p_h[:2*len(t):2,:] # (p_h[:2*len(t):2,:]+p_h[1:1+2*len(t):2,:])/2.
        p_v = p_v[:2*len(t):2,:] # (p_v[:2*len(t):2,:]+p_v[1:1+2*len(t):2,:])/2.
        p_hv = p_hv[:2*len(t):2,:] # (p_hv[:2*len(t):2,:]+p_hv[1:1+2*len(t):2,:])/2.
    
    xvalues = h5.freqs/1e6 if vs_freq else h5.channels
    xunit = "MHz" if vs_freq else "channel #"
    
    # Waterfall plots for debugging
    _suptitle_ = "%s: %s[%s] with FFT shift %d"%(filename,ant,h5.receivers[ant],fft_shift)
    figure(figsize=(16,8)); suptitle(_suptitle_)
    subplot(2,1,1); imshow(p_h/np.median(p_h,axis=0), vmin=0.99,vmax=1.01,cmap='jet', aspect='auto',extent=[xvalues[0],xvalues[-1],t[-1],t[0]], origin='upper')
    ylabel(r"$dP_H/<P_H>$\ntime [sec]"); colorbar()
    subplot(2,1,2); imshow(p_v/np.median(p_v,axis=0), vmin=0.99,vmax=1.01,cmap='jet', aspect='auto',extent=[xvalues[0],xvalues[-1],t[-1],t[0]], origin='upper')
    ylabel(r"$dP_V/<P_V>$\ntime [sec]"); xlabel("frequency [%s]"%xunit); colorbar()

    # Plots to identify RFI-free bits of spectrum
    _pol_lbl_ = lambda pol: "%s pol @ gain %s"%(pol,gains[0]["%s%s"%(ant,pol.lower())])
    _suptitle_ = "%s: %s[%s] with FFT shift %d\n%s %s; BWch,tau~(%.3fHz, %.3fsec)"%(filename,ant,h5.receivers[ant],fft_shift, _pol_lbl_("H"),_pol_lbl_("V"),h5.channel_width,dt)
    figure(figsize=(16,8)); suptitle(_suptitle_)

    subplot(2,1,1) # H & V sigma/mu spectra
    plot(xvalues, np.std(p_h,axis=0)/np.mean(p_h,axis=0))
    plot(xvalues, np.std(p_v,axis=0)/np.mean(p_v,axis=0))
    K = 1/np.sqrt(h5.channel_width*dt) # Expected radiometer scatter
    if cycle_50pct: # TODO: Work in progress - why is this necessary for cycle_50pct?
        K *= 40 # Perhaps requant gains wrong?
    plot(xvalues,K+0*h5.channels,'k,'); ylim(K/2.,2.6*np.max(xK)*K)
    ylabel(r"$\sigma/\mu$ []"); title("Complete spectrum")

    subplot(2,1,2) # HV power
    plot(xvalues, 10*np.log10(np.mean(p_hv,axis=0)))
    ylabel(r"HV [dB]")

    xlabel("Frequency [%s]"%xunit);

    # Identify pristine chunks of spectrum e.g. from the above
    M = max(int(10/dt),32) # Need something like 20/dt MHz to beat system noise?
    ch_chunks = [range(100+M*n,100+M*(n+1)) for n in range(1,(len(h5.channels)-200)//M)] # Omit 100 channels at both edges

    snr_h = [np.mean(np.std(p_h[:,C],axis=0)/np.mean(p_h[:,C],axis=0)) for C in ch_chunks]
    snr_v = [np.mean(np.std(p_v[:,C],axis=0)/np.mean(p_v[:,C],axis=0)) for C in ch_chunks]
    if (abs(1-np.nanmedian(snr_h)/np.nanmedian(snr_v)) > 1): # One pol is significantly worse than the other
        snr = np.nanmin([snr_h,snr_v],axis=0) # Min over frequency
    else:
        snr = 1/2**.5 * np.sqrt(np.asarray(snr_h)**2+np.asarray(snr_v)**2) # Average over frequency

    for x in xK: # K*(1+x) -- acceptable thresholds to filter out RFI
        snr_flags = snr<x*K # at most x% more than expected
        if (len(snr[snr_flags]) > 0):
            break
        else:
            print("WARNING: All chunks of this dataset fluctuate by more than %g%%. Bad RFI? Poor stability? Sun crossing the sidelobes?"%((x-1)*100))
    if (len(snr[snr_flags]) == 0):
        return None
    print("Proceeding with analysis on %d/%d channels passing RFI threshold of %g%%"%(np.count_nonzero(snr_flags),len(snr_flags),(x-1)*100))
    print(snr[snr_flags])
    
    hv = np.asarray([np.std(p_hv[:,C]/np.mean(p_hv[:,C],axis=0))-1 for C in ch_chunks]) # APH 06/2018 changed /np.std() to /np.mean()
    print(hv[snr_flags])
    hv_flags = hv<10*np.percentile(hv,10) # flag out chunks where HV varies greatly

    if np.all(~snr_flags[hv_flags]): # APH 06/2018 added this to avoid unnecessarily blocking good data?
        print("WARNING: skipping final flagging based on HV stability, since that leaves no data")
        ch_chunks = np.asarray(ch_chunks)[snr_flags]
    else:
        ch_chunks = np.asarray(ch_chunks)[snr_flags*hv_flags]
    print(ch_chunks)
    
    figure(figsize=(16,8)); suptitle(_suptitle_)
    subplot(2,1,1) # H & V sigma/mu spectra
    for ch in ch_chunks:
        plot(xvalues[ch], np.std(p_h[:,ch],axis=0)/np.mean(p_h[:,ch],axis=0))
        plot(xvalues[ch], np.std(p_v[:,ch],axis=0)/np.mean(p_v[:,ch],axis=0))
    plot(xvalues,K+0*h5.channels,'k,'); ylim(K/2.,3*K)
    ylabel(r"$\sigma/\mu$ []"); legend([_pol_lbl_("H"),_pol_lbl_("V")]); title("Pristine spectrum")

    subplot(2,1,2) # HV power
    for ch in ch_chunks:
        plot(xvalues[ch], 10*np.log10(np.mean(p_hv[:,ch],axis=0)))
    plot(xvalues,K+0*h5.channels,'k,');
    ylabel(r"HV [dB]")

    xlabel("Frequency [%s]"%xunit); 

    _suptitle_ = "%s: %s[%s] with FFT shift %d\n BWch,tau~(%.1fHz, %.3fsec)"%(filename,ant,h5.receivers[ant],fft_shift, h5.channel_width,dt)
    # Added figure to visually compare stability in selected channels against major RFI culprits
    _ch_chunks = [("clean",ch_chunks[-1]) # reference clean channel
                 ] + [(k,c) for (k,c) in [("GSM<40MHz>",h5.channels[np.abs(h5.freqs-940e6)<=20e6]), # GSM
                                          ("GSM<10MHz>",h5.channels[np.abs(h5.freqs-935e6)<=5e6]),  # GSM
                                          ("GSM<10MHz>",h5.channels[np.abs(h5.freqs-955e6)<=5e6]),  # GSM
                                          ("SSR-air",h5.channels[np.abs(h5.freqs-1090e6)<=2*2e6])]  # SSR-airTx; 2* opens this up to 1086MHz i.e. alias in UHF-band
                                          if (len(c)>0)]
    axes = subplots(len(_ch_chunks),1,figsize=(16,2+2*len(_ch_chunks)))[1]; suptitle("Compare to known RFI\n%s: %s"%(filename, ant))
    axes = np.atleast_1d(axes)
    for i,(key,ch) in enumerate(_ch_chunks):
        axes[i].plot(t, np.mean(p_h[:,ch],axis=1)/np.mean(p_h[:,ch]), label="H")
        axes[i].plot(t, np.mean(p_v[:,ch],axis=1)/np.mean(p_v[:,ch]), label="V", alpha=0.7)
        axes[i].set_ylabel(r"%s $\delta P/P$"%key); axes[i].legend(); axes[i].set_title("~%.f MHz"%(h5.freqs[ch].mean()/1e6))
    axes[-1].set_xlabel("time [sec]")

    # Time series H & V
    figure(figsize=(16,8)); suptitle(_suptitle_)
    subplot(2,1,1)
    for ch in ch_chunks:
        plot(t, np.mean(p_h[:,ch],axis=1)/np.mean(p_h[:,ch]))
    xlabel("time [sec]"); ylabel(r"$\delta P/P$ [linear]")
    subplot(2,1,2)
    for ch in ch_chunks:
        plot(t, np.mean(p_v[:,ch],axis=1)/np.mean(p_v[:,ch]))
    xlabel("time [sec]"); ylabel(r"$\delta P/P$ [linear]");
    
    # Generate the flags for spikes in time series
    t_A, t_B = None, None # Flags identifying the clean & spike windows respectively
    if t_spike_start is not None and t_spike_end is not None:
        AB = t_spike_end-t_spike_start
        if (t_spike_end < t[len(t)//2]): # Spike in first half of data, choose clean window after spike
            t_A = np.nonzero(np.abs(t-(t_spike_end+AB/2.))<=AB/2.) # Clean
            t_B = np.nonzero(np.abs(t-(t_spike_end-AB/2.))<=AB/2.) # Spike here
        else: # Spike in second half of data, choose clean window before spike
            t_A = np.nonzero(np.abs(t-(t_spike_start-AB/2.))<=AB/2.) # Clean
            t_B = np.nonzero(np.abs(t-(t_spike_start+AB/2.))<=AB/2.) # Spike here

    # Debug in case time-domain spikes are noticed
    if t_A is not None and t_B is not None:
        # Time domain
        figure(figsize=(16,8)); suptitle(_suptitle_)
        for ch in ch_chunks:
            plot(t[t_A], (np.mean(p_h[:,ch],axis=1)/np.mean(p_h[:,ch]))[t_A])
            plot(t[t_A], (np.mean(p_v[:,ch],axis=1)/np.mean(p_v[:,ch]))[t_A])
            plot(t[t_B], (np.mean(p_h[:,ch],axis=1)/np.mean(p_h[:,ch]))[t_B])
            plot(t[t_B], (np.mean(p_v[:,ch],axis=1)/np.mean(p_v[:,ch]))[t_B])
        xlabel("time [sec]"); ylabel(r"$\delta P/P$ []")

        # Spectral domain
        figure(figsize=(16,8)); suptitle(_suptitle_)
        plot(xvalues, 10*np.log10(np.mean(p_h[t_B,:].squeeze(),axis=0) / np.mean(p_h[t_A,:].squeeze(),axis=0)))
        plot(xvalues, 10*np.log10(np.mean(p_v[t_B,:].squeeze(),axis=0) / np.mean(p_v[t_A,:].squeeze(),axis=0)))
        ylim(-0.1,0.1)
        xlabel("Frequency [%s]"%xunit); ylabel("P(spike)/P(nospike) [dB#]")
  
    # Only use the data from start up to just before the spike
    if t_A is not None and t_B is not None:
        if (np.min(t_A)<np.min(t_B)): # Spike is after clean data
            t_Z = np.min(t_B)
            t = t[:t_Z]
            p_h = p_h[:t_Z,:]
            p_v = p_v[:t_Z,:]
        else: # Spike precedes clean data
            t_Z = np.min(t_A)
            t = t[t_Z:]
            p_h = p_h[t_Z:,:]
            p_v = p_v[t_Z:,:]

    # PSD of individual channel chunks, **normalized to fractional variation per channel**
    figure(figsize=(12,8)); suptitle(_suptitle_)
    subplot(2,1,1)
    for ch in ch_chunks:
        psd(np.mean(p_h[:,ch],axis=1)/np.mean(p_h[:,ch])-1, Fs=1/dt, NFFT=len(t))
    subplot(2,1,2)
    for ch in ch_chunks:
        psd(np.mean(p_v[:,ch],axis=1)/np.mean(p_v[:,ch])-1, Fs=1/dt, NFFT=len(t))

    # All good channels combined, un-normalized
    figure(figsize=(12,6)); suptitle(_suptitle_)
    P_h=np.take(p_h,ch_chunks,axis=1).reshape(len(t),np.prod(ch_chunks.shape))
    psd(np.mean(P_h,axis=1)-np.mean(P_h), Fs=1/dt, NFFT=len(t))
    P_v=np.take(p_v,ch_chunks,axis=1).reshape(len(t),np.prod(ch_chunks.shape))
    psd(np.mean(P_v,axis=1)-np.mean(P_v), Fs=1/dt, NFFT=len(t));       
    
    # Save combined time series data
    BW = P_h.shape[1] * abs(h5.freqs[1]-h5.freqs[0])
    np.savetxt("%s_%s.csv"%(filename[-13:-3],ant), [np.mean(P_h,axis=1), np.mean(P_v,axis=1)], delimiter=",",
               header="Autocorrelation stability from %s: %s with BW=%gMHz, dT=%gsec. Format: H, V [linear P_sys]"%(filename,ant,BW/1e6,dt))

    sfact = 2**-.5 if cycle_50pct else 1. # Scale factor for RMS & std, in case the samples are differences
    
    # RMS measurements in each identified good channel chunk individually
    axes = subplots(4, 1, figsize=(12,20))[1]; suptitle(_suptitle_)
    for i,(pol,p_t) in enumerate([("H",p_h), ("V",p_v)]):
        for ch in ch_chunks:
            p_tch = np.mean(p_t[:,ch],axis=1)/np.mean(p_t[:,ch])
            axes[2*i].plot(t, p_tch,'+', t, fit_avg(p_tch,T_interval/dt), '.'); axes[2*i].set_ylabel("Sampled power [linear]")
            axes[2*i].set_title("%s pol"%pol)
            axes[2*i+1].plot(t, 100*sliding_rms(p_tch,T_interval/dt)*sfact, label="ch ~%.f"%ch.mean());  axes[2*i+1].set_ylabel("RMS over %.f sec [%%]"%T_interval)
            axes[2*i+1].plot(t, sigma_spec+0*t, 'k--') # Spec limit
        axes[2*i+1].set_ylim(0,1.5*sigma_spec)
    axes[-1].legend()
    axes[-1].set_xlabel("time [sec]");

    # Measurements combined over all identified good channel chunks
    _suptitle_ = "%s: %s[%s] with FFT shift %d\n BWch,tau~(%.3fMHz, %.3fsec)"%(filename,ant,h5.receivers[ant],fft_shift, BW/1e6,dt)
    ret = []
    axes = subplots(4, 1, figsize=(12,20))[1]; suptitle(_suptitle_)
    for i,(pol,p_t) in enumerate([("H",P_h), ("V",P_v)]):
        p_t = np.mean(p_t,axis=1)/np.mean(p_t)
        ret.append(p_t)
        axes[2*i].plot(t, p_t,'+', t, fit_avg(p_t,T_interval/dt), '.'); axes[2*i].set_ylabel("Sampled power [linear]")
        result = 100*sliding_rms(p_t,T_interval/dt)*sfact
        axes[2*i].set_title("%s pol over %.f sec: 95th pct %.3f%%"%(pol,T_interval,np.percentile(result,95)))
        axes[2*i+1].plot(t, result);  axes[2*i+1].set_ylabel("RMS over %.f sec [%%]"%T_interval)
        axes[2*i+1].plot(t, sigma_spec+0*t, 'k--') # Spec limit
        axes[2*i+1].set_ylim(0,1.5*sigma_spec)
    axes[-1].set_xlabel("time [sec]");
    
    af = plot_allanvar(ret[0], dt=dt, sfact=sfact, label="H-pol");
    plot_allanvar(ret[1], dt=dt, sfact=sfact, label="V-pol", grid=True, figs=af)
    suptitle(_suptitle_)
    
    return ret

analyze = analyse # Alias


def get_fft_shift_and_gains(h5, channel=123, verbose=False):
    # in v4, fft_shift sensor values are stored per timestamp, but these never change
    try: # v4 after 2019?
        fft_shift = h5.sensor['wide_antenna_channelised_voltage_fft_shift'][0]
    except:
        try: # v4 up to 2019?
            fft_shift = h5.sensor['i0_antenna_channelised_voltage_fft_shift'][0]
        except: # v3 -- but these are always just the defaults?
            try:
                fft_shift = h5.file['TelescopeState'].attrs['cbf_fft_shift']
            except: # < v3
                fft_shift = "UNKNOWN"
    
    # Load requant gains from metadata. for timestamp[0] of each scan, assuming it never changes during a scan
    eq_gains = []
    for scan in h5.scans():
        if (len(h5.timestamps) < 2): continue # Some buggy observations have such tracks -- applied in troubleshoot() too
        eq_gains.append(dict(zip(["%sh"%a.name for a in h5.ants]+["%sv"%a.name for a in h5.ants],
                                 [-1 for a in h5.ants]+[-1 for a in h5.ants])))
        for port in eq_gains[-1].keys():
            try: # v4 after 2019?
                eq_gains[-1][port] = h5.sensor['wide_antenna_channelised_voltage_%s_eq'%port][0][channel]
            except:
                try: # v4 up to 2019?
                    eq_gains[-1][port] = h5.sensor['i0_antenna_channelised_voltage_%s_eq'%port][0][channel]
                except: # v3 -- but these are always just the defaults?
                    ports = [k for k in h5.sensor.keys() if "cbf_eq_coef" in k]
                    eq_gains[-1][port] = str(pickle.loads(h5.file[ports[0]][0][1]))
    
    if verbose:
        print("CBF FFT shift:%s %s" % (fft_shift, "" if isinstance(fft_shift,str) else bin(fft_shift)))
        print("CBF requantization (equalization) gains:\n%s" % eq_gains)
    
    return fft_shift, eq_gains
    
def troubleshoot(h5, ant, scans="track"):
    """ Troubleshooting fast gain stability reduction """
    filename = h5.name.split()[0].split(" ")[0] # "rdb files are different
    
    h5.select(ants=ant)
    h5.select(reset="", scans=scans, pol=("H","V"), corrprods="auto") # Only total power
    fft_shift, gains = get_fft_shift_and_gains(h5, verbose=True)
    
    # Correctly implemented detector output is Chi square, so sigma/mu = sqrt(2k)/k with k = 2*BW*tau
    K = np.sqrt(2)/np.sqrt(2*h5.channel_width*h5.dump_period)
    
    # Waterfall plots for debugging
    figure(figsize=(16,8))
    p_h = np.abs(h5.vis[:,:,0].squeeze())
    subplot(2,1,1); imshow(p_h/np.median(p_h,axis=0), vmin=0.99,vmax=1.01,cmap='jet', aspect='auto', origin='upper')
    ylabel(r"$dP_H/<P_H>$\ntime [dump]"); colorbar()
    p_v = np.abs(h5.vis[:,:,1].squeeze())
    subplot(2,1,2); imshow(p_v/np.median(p_v,axis=0), vmin=0.99,vmax=1.01,cmap='jet', aspect='auto', origin='upper')
    ylabel(r"$dP_V/<P_V>$\ntime [dump]"); xlabel("frequency"); colorbar()
    
    # Plot the time series power and get the mean spectra for all scans
    spectra_h, spectra_v, labels = [], [], []
    figure(figsize=(14,6))
    suptitle("%s: %s"%(filename.split("/")[-1], ant))
    for scan,eq_gains in zip(h5.scans(),gains):
        if (len(h5.timestamps) < 2): continue # Some buggy observations have such tracks
        # Compute the statistic using unbiased estimator (ddof=1) it's important for small samples
        x = h5.vis[:,:,0]; spectra_h.append(np.std(x,axis=0,ddof=1)/np.mean(x,axis=0))
        plot(h5.timestamps, x.mean(axis=1), label="H")
        x = h5.vis[:,:,1]; spectra_v.append(np.std(x,axis=0,ddof=1)/np.mean(x,axis=0))
        plot(h5.timestamps, x.mean(axis=1), label="V")
        labels.append("scan %d"%(scan[0]))
    legend()
    fft_shift, B, t, K = [fft_shift], [h5.channel_width], [h5.dump_period], [K]*len(spectra_h)

    # sigma/mu over detected time series, per channel
    figure(figsize=(14,6))
    suptitle("%s: %s\n BWch,tau~%s Hz,sec; FFT shift~%s\nDetector anomaly"%(filename.split("/")[-1],ant,set(zip(B,t)),set(fft_shift)))
    for i,spectra in enumerate([spectra_h,spectra_v]):
        subplot(1,2,1+i)
        _P = 'hv'[i]
        for _K,_s,_l,_g in zip(K,spectra,labels,gains):
            _g = [_g[k] for k in _g.keys() if k.endswith(_P)]
            plot(_s, label=_l+", gains %s"%(_g))
            plot(0*_s+_K, 'k-')
        legend(); xlabel("channel [#]"); ylabel(r"$\sigma/\mu$"); title(_P)
        ylim(0,1.1*np.nanpercentile(spectra,99))
        cleanchans = range(len(_s))
        plot(cleanchans, 0*np.asarray(cleanchans), 'mo')


################################################# Command Line Interface #################################################
if __name__ == "__main__":
    import optparse
    # Parse command-line opts and arguments
    parser = optparse.OptionParser(usage="%prog [opts] <HDF5 file>",
                                   description="This processes an HDF5 dataset and generates figures.")
    parser.add_option("--troubleshoot", action='store_true',
                      help="Troubleshoot diagnotics for a specified antenna, rather than actual processing.")
    parser.add_option("--hackL", action='store_true',
                      help="Open the data file as if it was recorded with an L-band digitiser with the 2016 hacked band select filter")
    parser.add_option("--channels", type='string', default=None,
                      help="Use this to select channels for processing, if necessary, as per katdal convention (default = %default)")
    parser.add_option("--ant", type='string', default=None,
                      help="Specific antenna to run analysis for (default = %default)")
    parser.add_option("--spike-start", type='float', default=None,
                      help="Start of spike in time series to be omitted, in seconds (default = %default)")
    parser.add_option("--spike-end", type='float', default=None,
                      help="End of spike in time series to be omitted, in seconds (default = %default)")
    parser.add_option("-T", "--interval", type='float', default=5,
                      help="Time duration of sliding windows to evaluate over, in seconds (default = %default)")
    parser.add_option("-X", "--sigma_spec", type='float', default=0.10,
                      help="The reference limit in percent to indicate on the final figures (default = %default)")
    parser.add_option("--freq", action='store_true',
                      help="Add this switch to have figures displayed vs. frequency rather than channel number")
    parser.add_option("--xK", type='string', default='1.03,1.05,1.1,1.15',
                      help="Use this to set acceptable thresholds for exces above expected threshold (default = %default)")
    parser.add_option("--nd50", action='store_true',
                      help="Add this switch to process the difference between consecutive time samples, e.g. when Noise Diode toggles at 50% duty cycle over two accumulation intervals")

    (opts, args) = parser.parse_args()
    if len(args) != 1 or not (args[0].endswith('.h5') or args[0].endswith('.rdb')):
        raise RuntimeError('Please specify a single HDF5 file as argument to the script')

    filename = args[0]
    ant = opts.ant
    t_spike_start, t_spike_end = opts.spike_start, opts.spike_end
    
    h5 = open_dataset(filename, ref_ant=ant, hackedL=opts.hackL)
    print(h5)
    print(h5.receivers)
    
    if not opts.troubleshoot:
        get_fft_shift_and_gains(h5, verbose=True)
        if ant:
            xK = [float(x) for x in opts.xK.split(",")]
            p_t = analyse(h5, ant, t_spike_start, t_spike_end, eval(opts.channels) if opts.channels else None, opts.freq, opts.interval, opts.sigma_spec, opts.nd50, xK)
        else:
            for ant in h5.ants:
                p_t = analyse(h5, ant.name, t_spike_start, t_spike_end, eval(opts.channels) if opts.channels else None, opts.freq, opts.interval, opts.sigma_spec, opts.nd50, xK)
    else:
        if ant:
            troubleshoot(h5, ant)
        else:
            for ant in h5.ants:
                troubleshoot(h5, ant)

    plt.show()
