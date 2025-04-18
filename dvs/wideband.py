"""
    Analysis for the "wide band" spectrum analyser measurements.
    
    Example of use:
        process_wbg_set(r"C:\Work\saraoProjects\dvs\SKA119\l1_data\WBG8GHz_SKA119_11092024", "SPFB2", flim=(0,4e9))
        plt.suptitle("WBG8GHz_SKA119_11092024 | SPFB2 SN-013")
        process_wbg_set(r"C:\Work\saraoProjects\dvs\SKA119\l1_data\WBG8GHz_SKA119_25092024", "SPFB2", flim=(0,4e9))
        plt.suptitle("WBG8GHz_SKA119_25092024 | SPFB2 SN-012")
"""
import pylab as plt
import numpy as np
import os


dB2lin = lambda dB: 10**(dB/10)
lin2dB = lambda lin: 10*np.log10(lin)


def load_rs_traces(filename):
    """ Load data from a "WBG" measurement recorded with a R&S spectrum analyser.
        
        @param filename: ASCII exported trace file from the spectrum analyzer.
        @return: (xdata, ydata, header) with data first dimension corresponding to "x" and header a dictionary. 
    """
    # Load and interpret the header - columns are: x, y, unit, empty_column[, ...]
    with open(filename, encoding="UTF-8-SIG") as file: # SIG handles potential opening bytes that identify this as UTF-8 on Windows
        header = []
        for line in file:
            if (line.find(",") <= 0): # An empty line (or one that doesn't include comma separated fields) means end of header
                break
            header.append(line.split(","))
        # The first line after the empty line contains the data labels, followed by the data
        header.append(file.readline().split(","))
    labels = header[-1]
    traces = header[0][1::4]
    xlabels, ylabels = labels[0::4], labels[1::4]
    
    # Load the data - columns are x, y, empty_column, empty_column[, ...]
    data = np.loadtxt(filename, encoding="UTF-8-SIG", delimiter=",", skiprows=len(header)+2,
                      usecols=[c for c,l in enumerate(labels) if len(l.strip())>0])
    xdata, ydata = data[:,0::2], data[:,1::2]
    
    def str2num(v):
        try:
            return float(v)
        except:
            return v 
    header = {a:str2num(b) for a,b,*_ in header[1:-1]} # Omit labels & traces
    header.update({"filename":filename, "xlabels":xlabels, "ylabels":ylabels, "traces":traces})
    return (xdata, ydata, header)


def load_dig_spectra(filenames, f_sample, NYQ=2):
    """ Load DIGITISER spectrometer data from a local file.
        
        @param filenames: one or more filenames containing the digitiser spectrum data
        @param f_sample: digitiser sample rate [Hz]
        @param NYQ: Nyquist zone sampled by the digitiser (default 1).
        @return: (freqs, spectrum0, ...) in [Hz] and [complex power, dBcounts] """
    freqs = None
    spectra = []
    for fn in np.atleast_1d(filenames):
        spec = np.loadtxt(fn, delimiter=",")
        spectra.append(spec)
        if (freqs is None):
            freqs = np.linspace(NYQ-1, NYQ, len(spec))*f_sample/2
            # TODO: interpret NYQ
    return [freqs] + spectra


class WBGDataset(object):
    def __init__(self, freq, pol0_off, pol0_on, pol1_off, pol1_on, pols="HV", header=None):
        """
            @param freq: frequencies [Hz] (or as per header["xlabel"])
            @param pol?_off: magnitudes with ND OFF, to match frequencies [dBm] (or as per header["ylabels"])
            @param pol?_on: magnitudes with ND ON, to match frequencies [dBm] (or as per header["ylabels"])
            @param pols: labels for pol0 & pol1 (default "HV")
            @param instrument: (same units as pol...)
            @param header: dictionary with at least 'xlabel' & 'ylabel'->strings.
        """
        self.pols = pols
        self.freq = freq
        self.off = [pol0_off, pol1_off]
        self.on = [pol0_on, pol1_on]
        self.header = {} if (header is None) else header
        self.RBW = header.get("RBW", np.diff(freq).mean())
        self.off_maxhold = None
        self.on_maxhold = None
    
    def de_embedded(self, instr_freq, instr_mag0, instr_mag1):
        """ Remove the instrument's contribution from the data.
            
            @param instr_freq: frequencies to match or overlap with .freq (same units & overlapping
            @param instr_mag0,1: the instrument's contribution (same units as .on & .off)
            @return: WBGDataset, with 'on' & 'off' values with the instrumental contribution removed
        """
        # Interpolate as necessary to match frequencies
        instr_mag = np.array([np.interp(self.freq, instr_freq, instr_mag0),
                              np.interp(self.freq, instr_freq, instr_mag1)])
        
        if ("dB" in self.header["ylabel"]):
            instr_mag = dB2lin(instr_mag)
            de_embed = lambda dB: lin2dB( dB2lin(dB) - instr_mag )
        else:
            de_embed = lambda lin: lin - instr_mag
        d_on = de_embed(np.asarray(self.on))
        d_off = de_embed(np.asarray(self.off))
        
        return WBGDataset(self.freq, d_off[0], d_on[0], d_off[1], d_on[1], self.pols, self.header)
    
    def scale_to_T(self, T_ND_model):
        """ Scale the data to temperature, using the measured ND ON-OFF and the given T_ND_model data.
            
            @param T_ND_model: an instance of katsemodels.Rec_T_ND
            @return: (freq [Hz], Tsys_pol0, Tsys_pol1 [K]) """
        Tnd_p0 = T_ND_model.nd["HH"](self.freq/1e6)
        Tnd_p1 = T_ND_model.nd["VV"](self.freq/1e6)
        
        on, off = self.on, self.off
        if ("dB" in self.header["ylabel"]):
            on = dB2lin(np.asarray(on))
            off = dB2lin(np.asarray(off))
            
        Y_factor = on / off
        Tsys_p0 = Tnd_p0/(Y_factor[0]-1)
        Tsys_p1 = Tnd_p1/(Y_factor[1]-1)
        
        return (self.freq, Tsys_p0, Tsys_p1)
    
    @classmethod
    def load(cls, root_folder, pols="HV"):
        """ Loads a dataset that was generated using the standard "wizard"
            
            @return: WBGDataset """
        # This dataset is generated using a "wizard", so the file format is exactly know
        #   pol* is [[avg(RMS), max_hold(RMS)] by freq]
        freq, pol0_off, header = load_rs_traces(root_folder+f"/{pols[0]}polND_OFF.csv")
        freq, pol0_on, header = load_rs_traces(root_folder+f"/{pols[0]}polND_ON.csv")
        freq, pol1_off, header = load_rs_traces(root_folder+f"/{pols[1]}polND_OFF.csv")
        freq, pol1_on, header = load_rs_traces(root_folder+f"/{pols[1]}polND_ON.csv")
        header.update({"xlabel":header['xlabels'][0], "ylabel":header['ylabels'][0]})
        dataset = WBGDataset(freq[:,0], pol0_off[:,0], pol0_on[:,0], pol1_off[:,0], pol1_on[:,0], pols=pols, header=header)
        # Some extras for this kind of dataset
        dataset.on_maxhold = [pol0_on[:,1], pol1_on[:,1]]
        dataset.off_maxhold = [pol0_off[:,1], pol1_off[:,1]]
        
        try:
            freq_i, pol0_i, header_i = load_rs_traces(root_folder+f"/{pols[0]}pol_Instrument.csv")
            freq_i, pol1_i, header_i = load_rs_traces(root_folder+f"/{pols[1]}pol_Instrument.csv")
            dataset = dataset.de_embedded(freq_i, pol0_i, pol1_i)
        except IOError as e:
            print("WARNING: Unable to de-embed the spectrum analyser & cable response!", e)
        
        return dataset

    @classmethod
    def load_dig(cls, root_folder, f_sample, NYQ=2, pols="HV"):
        """ Load a digitiser spectrometer dataset as if it's a WBGDataset.
            NB: The files in root_folder must be sorted in this sequence: (pols[0]_off,pols[0]_on, pols[1]_off,pols[1]_on)
            
            @return WBGDataset """
        # The files in the folder are assumed to be sorted in this sequence: (H_off,H_on, V_off,V_on)
        files = sorted(os.listdir(root_folder))
        files = [root_folder+"/"+f for f in files if ("DIFF" not in f)] # Remove the "HV" cross correlation files
        freq, h_off = load_dig_spectra(files[0], f_sample, NYQ)
        freq, h_on = load_dig_spectra(files[1], f_sample, NYQ)
        freq, v_off = load_dig_spectra(files[2], f_sample, NYQ)
        freq, v_on = load_dig_spectra(files[3], f_sample, NYQ)
        header = {"xlabel":"Frequency [Hz]", "ylabel":"Power [dBcounts]"}
        dataset = WBGDataset(freq, h_off, h_on, v_off, v_on, pols=pols, header=header)
        return dataset


def band_defs(band_ID):
    """ Returns the appropriate frequency endpoints and the related MeerKAT WBG limit lines
        for the specified frequency band.
        GTsys_ref represents 18K plus 56dB gain, in linear scale.
        
        @param band_ID: 'u'|'l'|'B1'...
        @return: freq_band, fracnd_limits, GTsys_ref, band_mask """
    if ('l' in band_ID) or ("B2" in band_ID):
        ulim=-124.3+10*np.log10(3e6)
        mask_f = [0,200e6,420e6,2150e6,2900e6,3600e6]
        mask_a = [ulim-21, ulim-21, ulim, ulim, ulim-21, ulim-21]
        if ('l' in band_ID): # MK L 
            band_freq = (900e6, 1670e6)
            nd_lim = (0.5,1.5)
        else: # SPFB2
            band_freq = (950e6, 1760e6)
            nd_lim = (0.05,0.13)
    elif ('u' in band_ID): # MK UHF
        ulim=-122.7+10*np.log10(3e6)
        mask_f = [0,100e6,300e6,1200e6,np.nan,1610e6,3600e6]
        mask_a = [ulim-13, ulim-13, ulim, ulim,np.nan, ulim-13, ulim-13]
        band_freq = (580e6,1050e6)
        nd_lim = (0.5,1.5)
    
    GTsys_ref = 10**5.6 * 18 # 18K + 56dB
    
    return band_freq, nd_lim, GTsys_ref, (mask_f,mask_a)


def process_wbg_set(dataset, band_ID, flim=None, figsize=None):
    """ Process a set of H & V pol measurements with ND OFF and ON.
        Generates a standard figure to summarise the results.
        
        @param dataset: either a WBGDataset or a descriptor that can be padded to WBGDataset.load()
        @param flim: frequency limits for display only, as (f_start,f_stop) in same units as dataset.
        @return: WBGDataset
    """
    freq_band, nd_lims, GTsys_ref, band_mask = band_defs(band_ID)
    nlim = nd_lims[-1] - nd_lims[0]; nlim = (np.mean(nd_lims)-2*nlim, np.mean(nd_lims)+2*nlim)
    dataset = dataset if isinstance(dataset, WBGDataset) else WBGDataset.load(dataset)
    subset_mask = (dataset.freq>=freq_band[0]) & (dataset.freq<=freq_band[-1])
    
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(3, 2, figure=fig)
    
    axs = [fig.add_subplot(gs[i, :]) for i in range(gs.nrows-1)]
    for off,on,pol in zip(dataset.off,dataset.on,dataset.pols):
        axs[0].plot(dataset.freq, off, label=pol+"_OFF")
        axs[0].plot(dataset.freq, on, label=pol+"_ON")
        # Ratio of T_ND/Tsys
        axs[1].plot(dataset.freq, dB2lin(on)/dB2lin(off) - 1, label=pol)
    kB = 1.38e-23 # Boltzmann's constant
    axs[0].plot(dataset.freq, np.full_like(dataset.freq,lin2dB(kB*GTsys_ref*dataset.RBW)+30), 'm--', label="Tsys_ref")
    axs[0].plot(band_mask[0], band_mask[1], 'k-')
    axs[0].set_ylabel(dataset.header["ylabel"]); axs[0].legend()
    axs[1].set_ylabel("TND/Tsys [frac]"); axs[1].legend()
    axs[1].hlines(nd_lims, np.min(dataset.freq), np.max(dataset.freq), 'r')
    axs[1].set_ylim(*nlim)
    axs[1].set_xlabel(dataset.header["xlabel"])
    freq_subset = dataset.freq[subset_mask]
    freq_subset = (np.min(freq_subset), np.max(freq_subset))
    flim = flim if ( flim is not None) else freq_subset
    for ax in axs:
        ax.grid(True); ax.sharex(axs[0])
        ax.vlines(freq_subset, *ax.get_ylim(), 'm')
        ax.set_xlim(*flim)
    
    # RFI should show up in "max_hold"-"RMS" ~ 9 sigma ~10dB
    if False: # dataset.off_maxhold is not None:
        for offm,off, onm,on in zip(dataset.off_maxhold,dataset.off, dataset.on_maxhold,dataset.on):
            axs[1].plot(dataset.freq, offm-off -10, 'k-', alpha=0.7)
            axs[1].plot(dataset.freq, onm-on -10, 'k-', alpha=0.7)
    
    # Statistics only over subset of frequency!
    axs = [fig.add_subplot(gs[-1, i]) for i in range(2)]
    bins = np.linspace(*nlim,30)
    for ax,off,on,pol in zip(axs,dataset.off,dataset.on,dataset.pols):
        ax.hist((dB2lin(on)/dB2lin(off))[subset_mask] - 1, bins=bins, density=True, cumulative=True)
        ax.set_xlabel(f"{pol} TND/Tsys [frac]")
        ax.grid(True)
        ax.fill_between([bins[0],nd_lims[0]], [0.05,0.05], [1,1], color='r', alpha=0.3)
        ax.fill_between([nd_lims[-1],bins[-1]], [0,0], [0.95,0.95], color='r', alpha=0.3)

    return dataset
