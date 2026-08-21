'''
    Construct DVS SKA Digitiser calibration files.
    
    TODO: There are single point drop-outs in the WBG8GHz_DIG_QTP spectra at ~413MHz, ~1995MHz, ~4492MHz, ~5805MHz, ~6875MHz that seems to be spectrum analyser artefacts!
     
    @author aph@sarao.ac.za
'''
import pylab as plt
import numpy as np

from dvs.wideband import load_rs_traces, load_dig_spectra
from analysis import katsepnt as ksp


def convert_dig_spectra_to_gain(PN, SN, f_sample=1712e6, NYQ=1, debug=False):
    _, ax = plt.subplots(1,1); plt.suptitle(f"{PN} SN-{SN}")
    fig_p, axs_p = plt.subplots(2,1, sharex=True); plt.suptitle(f"{PN} SN-{SN}")
    for pol in 'HV':
        freq, pp = [], []
        
        spec_files = ksp.find_files(f"M0_{SN}_{pol}_Pol*.csv", root_dir=f"./{PN}")
        for fn in spec_files:
            freq, psd = load_dig_spectra(fn, f_sample=f_sample, NYQ=NYQ)
            TS = fn.split('.')[-2].split('_')[-1]
            axs_p[0].plot(freq, psd, label=TS)
            pp.append(10**(np.array(psd)/10)) # dB->linear
        axs_p[0].legend()
        pp = 10*np.log10(np.mean(pp, axis=0)) # Digitiser mean = dB(input power*dig_gain)
        
        # Reference to de-embed from digitiser output power spectrum: this was recorded using the exact same set-up as the T0015-002-noise_spectra
        freq_ref, psd_ref, header_ref = load_rs_traces(f"./WBG8GHz_DIG_QTP/{pol}polND_OFF.csv")
        # Interpolate as necessary to match digitiser frequencies
        instr_mag = np.interp(freq, freq_ref[:,0], psd_ref[:,0]) # dB RMS
        axs_p[1].plot(freq_ref, psd_ref, {'H':'-', 'V':'--'}[pol])
        axs_p[1].plot(freq, instr_mag, '|', alpha=0.3)
        
        # Equivalent power at input - linear power with instrument's calibrated gain (assumed unit gain)
        dig_gain = pp - instr_mag
        
        savefile = f"./{PN}_{SN}_{pol}pol.csv"
        header = f"{PN} SN-{SN} wideband noise power spectra and gain.\n" +\
                 f"Digitiser datasets=M0_{SN}_{pol}_Pol_sw_spec_dump_?.csv, digitiser spectrometer output, average of {len(spec_files)}\n" +\
                 f"REF datasets=WBG8GHz_DIG_QTP_{pol}pol_ND_OFF.csv, spectrum analyser output from same input.\n" +\
                  "Input signal is NoiseWave model NW6G-MI with 13dB attenuation, fed through a 3dB divider to H & V inputs.\n\n" +\
                  "Frequency [Hz],Channel power [dBX],Gain [dB]"
        np.savetxt(savefile, np.c_[freq, pp, dig_gain], fmt='%2.8f', header=header, delimiter=",")
        
        ax.plot(freq, dig_gain, label=pol)
    ax.legend()
    if not debug: plt.close(fig_p.number)


if __name__ == "__main__":
    convert_dig_spectra_to_gain("T0015-002", "L5", f_sample=1712e6, NYQ=1, debug=True)
    plt.show()
