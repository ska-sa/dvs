'''
    Construct DVS SKA Digitiser calibration files.
    
    TODO: for B1 digitiser there's a spike in the gains at ~400MHz that seems to be from the noise source - not sure why it's not present in the Digitiser output?
     
    @author aph@sarao.ac.za
'''
import pylab as plt
import numpy as np

kB = 1.38e-23 # Boltzmann's constant
T0 = 290 # Thermodynamic reference temperature


##### Copied from dvs/wideband.py
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
    data = np.loadtxt(filename, encoding="UTF-8-SIG", delimiter=",", skiprows=len(header)+1,
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


##### Copied from dvs/wideband.py
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


def convert_b1dig_spectra_to_gain():
    fig = plt.figure()
    fig_p = plt.figure()
    for pol in 'HV':
        plt.figure(fig_p.number)
        freq, pp = [], []
        for TS in ['13h58m34s','13h59m3s','13h59m18s']:
            freq, psd = load_dig_spectra(f"./M0_L5_{pol}_Pol_sw_spec_dump_{TS}.csv", f_sample=1712e6, NYQ=1)
            plt.plot(freq, psd, label=TS)
            pp.append(10**(np.array(psd)/10)) # dB->linear
        plt.legend()
        pp = 10*np.log10(np.mean(pp, axis=0)) # Digitiser mean = dB(input power*dig_gain)
        
        # Reference to de-embed from digitiser output power spectrum: this was recorded using the exact same set-up as the T0015-002-noise_spectra
        freq_ref, psd_ref, header_ref = load_rs_traces(f"./WBG8GHz_DIG_QTP/{pol}polND_OFF.csv")
        # Interpolate as necessary to match digitiser frequencies
        instr_mag = np.interp(freq, freq_ref[:,0], psd_ref[:,0]) # dB RMS
        # Equivalent power at input - linear power with instrument's calibrated gain (assumed unit gain)
        dig_gain = pp - instr_mag
        
        savefile = "./T0015-002_%spol.csv"%pol
        header = f"T0015-002 (E3301-SN05 with B1 RFCU) wideband noise power spectra and gain.\n" +\
                 f"Digitiser datasets=M0_L5_{pol}_Pol_sw_spec_dump_?.csv, digitiser spectrometer output, average of 3\n" +\
                 f"REF datasets=WBG8GHz_DIG_QTP_{pol}pol_ND_OFF.csv, spectrum analyser output from same input. 3\n" +\
                  "Input signal is NoiseWave model NW6G-MI with 13dB attenuation, fed through a 3dB divider to H & V inputs.\n\n" +\
                  "Frequency [Hz],Channel power [dBX],Gain [dB]"
        np.savetxt(savefile, np.c_[freq, pp, dig_gain], fmt='%2.8f', header=header, delimiter=",")
        
        plt.figure(fig.number); plt.plot(freq, dig_gain, label=pol)
    plt.figure(fig.number); plt.legend()
    plt.figure(fig_p.number); plt.legend()


if __name__ == "__main__":
    convert_b1dig_spectra_to_gain()
    plt.show()
