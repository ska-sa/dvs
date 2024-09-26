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
    xlabels, ylabels = header[-1][0::4], header[-1][1::4]
    
    # Load the data - columns are x, y, empty_column, empty_column[, ...]
    data = np.loadtxt(filename, encoding="UTF-8-sig", delimiter=",", skiprows=len(header)+2,
                      usecols=[c for c,l in enumerate(labels) if len(l.strip())>0])
    xdata, ydata = data[:,0::2], data[:,1::2]
    
    header = {"filename":filename, "xlabels":xlabels, "ylabels":ylabels, "traces":traces}
    return (xdata, ydata, header)


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
    
    def de_embedded(self, instr_freq, instr_mag):
        """ Remove the instrument's contribution from the data.
            
            @param instr_freq: frequencies to match or overlap with .freq (same units & overlapping
            @param instr_mag: the instrument's contribution (same units as .on & .off)
            @return: WBGDataset, with 'on' & 'off' values with the instrumental contribution removed
        """
        # Interpolate as necessary to match frequencies
        instr_mag = np.interp(self.freq, instr_freq, instr_mag)
        # Define de-embedding formulas
        if ("dB" in self.header["ylabel"]):
            instr_mag = dB2lin(instr_mag)
            de_embed = lambda dB: lin2dB( dB2lin(dB) - instr_mag )
        else:
            de_embed = lambda lin: lin - instr_mag
        if (len(np.shape(self.on[0])) > len(np.shape(instr_mag))):
            instr_mag = np.c_[instr_mag, instr_mag]
        # De-embed
        d_on = [de_embed(d) for d in self.on]
        d_off = [de_embed(d) for d in self.off]
        
        return WBGDataset(self.freq, d_off[0], d_on[0], d_off[1], d_on[1], self.pols, self.header)
    
    @classmethod
    def load(cls, root_folder, pols="HV"):
        # This dataset is generated using a "wizard", so the file format is exactly know
        freq, pol0_off, header = load_rs_traces(root_folder+f"/{pols[0]}polND_OFF.csv")
        freq, pol0_on, header = load_rs_traces(root_folder+f"/{pols[0]}polND_ON.csv")
        freq, pol1_off, header = load_rs_traces(root_folder+f"/{pols[1]}polND_OFF.csv")
        freq, pol1_on, header = load_rs_traces(root_folder+f"/{pols[1]}polND_ON.csv")
        i_freq, i_mag, _ = freq[:,0], freq[:,0]-np.inf, None # TODO!
        dataset = WBGDataset(freq[:,0], pol0_off[:,0], pol0_on[:,0], pol1_off[:,0], pol1_on[:,0], pols=pols,
                             header={"xlabel":header['xlabels'][0], "ylabel":header['ylabels'][0]}).de_embedded(i_freq, i_mag)
        # Some extras for this kind of dataset
        dataset.on_max = [pol0_on[:,1], pol1_on[:,1]]
        dataset.off_max = [pol0_off[:,1], pol1_off[:,1]]
        return dataset


def band_defs(band_ID):
    """ Returns the appropriate frequency endpoints and the related MeerKAT WBG limit lines
        for the specified frequency band.
        
        @param band_ID: 'u'|'l'|'B1'...
        @return: freq_band, fracnd_limits, band_mask """
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
    return band_freq, nd_lim, (mask_f,mask_a)


def process_wbg_set(dataset, band_ID, flim=None, figsize=None):
    """ Process a set of H & V pol measurements with ND OFF and ON.
        Generates a standard figure to summarise the results.
        @param dataset: either a WBGDataset or a descriptor that can be padded to WBGDataset.load()
        @param flim: frequency limits for display only, as (f_start,f_stop) in same units as dataset.
    """
    freq_band, nd_lim, band_mask = band_defs(band_ID)
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
    axs[0].plot(band_mask[0], band_mask[1], 'k-')
    axs[0].set_ylabel(dataset.header["ylabel"]); axs[0].legend()
    axs[1].set_ylabel("TND/Tsys [frac]"); axs[1].legend()
    axs[1].hlines(nd_lim, np.min(dataset.freq), np.max(dataset.freq), 'r')
    axs[1].set_ylim(nd_lim[0]/3,nd_lim[-1]*3)
    axs[1].set_xlabel(dataset.header["xlabel"])
    freq_subset = dataset.freq[subset_mask]
    freq_subset = (np.min(freq_subset), np.max(freq_subset))
    flim = flim if ( flim is not None) else freq_subset
    for ax in axs:
        ax.grid(True); ax.sharex(axs[0])
        ax.vlines(freq_subset, *ax.get_ylim(), 'm')
        ax.set_xlim(*flim)
    
    if False: # RFI should show up in "max_hold"-"RMS" ~ 9 sigma ~10dB
        try:
            for offm,off, onm,on in zip(dataset.off_max,dataset.off, dataset.on_max,dataset.on):
                axs[1].plot(dataset.freq, offm-off -10, 'k-', alpha=0.7)
                axs[1].plot(dataset.freq, onm-on -10, 'k-', alpha=0.7)
        except AttributeError:
            pass
    
    # Statistics only over subset of frequency!
    axs = [fig.add_subplot(gs[-1, i]) for i in range(2)]
    bins = np.linspace(nd_lim[0]/3,nd_lim[-1]*3,30)
    for ax,off,on,pol in zip(axs,dataset.off,dataset.on,dataset.pols):
        ax.hist((dB2lin(on)/dB2lin(off))[subset_mask] - 1, bins=bins, density=True, cumulative=True)
        ax.set_xlabel(f"{pol} TND/Tsys [frac]")
        ax.grid(True)
        ax.fill_between([bins[0],nd_lim[0]], [0.05,0.05], [1,1], color='r', alpha=0.3)
        ax.fill_between([nd_lim[-1],bins[-1]], [0,0], [0.95,0.95], color='r', alpha=0.3)
    
