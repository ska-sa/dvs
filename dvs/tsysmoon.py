#!/usr/bin/python
"""
    Analysis of single dish "hot load & cold sky" measurement to drive Tsys and T_noisediode.
    Data is generated using <observations/skyloadcal.py>
    
    Typical use 1:
        python tsysmoon.py --ants "m017,m023" /var/data/132598363.h5
    which is equivalent to
        import tsysmoon
        tsysmoon.process("/var/data/132598363.h5", ants=["m017","m023"])
    
    @author aph@sarao.ac.za
"""
# Originally migrated from [https://github.com/ska-sa/katsdpscripts/blob/master/katsdpscripts/RTS/diodelib.py] to RTS notebook 10/2015
# Migrated from http://localhost:8891/notebooks/systems-analysis/sandbox/adriaan/_OLD_NOTEBOOKS_052019/Tsys%20Moon%20Checks-UHF-final.ipynb to this script 02/2025
import numpy as np
import scipy as sp
import time, ephem
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from . import util, modelsroot
from analysis import katsemodels as models
import scape
import logging
logger = logging.getLogger("tsysmoon")

# Noise Diode Models that are not yet "deployed in the telescope"
nd_models_folder = modelsroot + '/noise-diode-models'


def get_nd_on_off(ds, buff=2,log=None): 
    on = ds.sensor['Antennas/%s/nd_coupler'%(ds.ants[0].name)]
    off = ~ds.sensor['Antennas/%s/nd_coupler'%(ds.ants[0].name)]
    n_on = np.tile(False,on.shape[0])
    n_off = np.tile(False,on.shape[0])
    if not any(on):
        log.warn('No noise diode fired during track')
    else:
        n_on[on.nonzero()[0][buff:len(n_on)-buff]] = True
        n_off[off.nonzero()[0][buff:len(n_on)-buff]] = True
    return n_on,n_off


def plot_Tsys_eta_A(freq,Tsys,eta_A,TAc,Ku=False,Tsys_std=None,ant = '', file_base='.', plot_speclines=False):
    fig = plt.figure(figsize=(20,5))
    pols = ['v','h']
    for p,pol in enumerate(pols) : 
        fig.add_subplot(1,2,p+1)  
        plt.title('%s $T_{sys}/eta_{A}$: %s pol: %s'%(ant,str(pol).upper(),file_base))
        plt.ylabel("$T_{sys}/eta_{A}$ [K]")
        plt.xlabel('f [MHz]')
        #if p == ant_num * 2 -1: plt.ylabel(ant)
        if Tsys_std[pol] is not None :
            plt.errorbar(freq/1e6,Tsys[pol],Tsys_std[pol],color = 'b',linestyle = '.',label='Measurement')
        T_e = Tsys[pol]/eta_A[pol]
        plt.plot(freq/1e6,T_e,'b.',label='Measurement: Y-method')
        plt.axhline(np.mean(T_e),linewidth=2,color='k',label='Mean: Y-method')
        if not(Ku): plt.plot(freq/1e6,TAc[pol]/eta_A[pol],'c.',label='Measurement: ND calibration')
        plt.ylim(np.percentile(T_e, 1), np.percentile(T_e, 99))
        if plot_speclines and (freq.min() < 2090e6): # MeerKAT UHF & L-band specifications
            D = 13.5
            Ag = np.pi* (D/2)**2 # Antenna geometric area
            spec_Tsys_eta = np.zeros_like(freq)
            plt.ylim(15,50)
            #plt.xlim(900,1670)
            spec_Tsys_eta[freq<1420e6] =  42 # [R.T.P095] == 220
            spec_Tsys_eta[freq>=1420e6] =  46 # [R.T.P.096] == 200
            plt.plot(freq/1e6, spec_Tsys_eta,'r',linewidth=2,label='PDR Spec')
            plt.plot(freq/1e6,np.interp(freq/1e6,[900,1670],[(64*Ag)/275.0,
                    (64*Ag)/410.0]),'g',linewidth=2,label="275-410 m^2/K at Receivers CDR")
            plt.grid()
            plt.legend(loc=2,fontsize=12)
            
    return fig   

def plot_nd(freq,Tdiode,nd_temp,ant = '', file_base=''): 
    fig = plt.figure(figsize=(20,5))
    pols = ['v','h']
    for p,pol in enumerate(pols) : 
        fig.add_subplot(1,2,p+1) # 
        plt.title('%s Coupler Diode: %s pol: %s'%(ant,str(pol).upper(),file_base))
        plt.ylim(0,50)
        plt.ylabel('$T_{ND}$ [K]')
        #plt.xlim(900,1670)
        plt.xlabel('f [MHz]')
        #plt.ylabel(ant)
        plt.axhspan(14, 35, facecolor='g', alpha=0.5)
        plt.plot(freq/1e6,Tdiode[pol],'b.',label='Measurement: Y-method')
        plt.plot(freq/1e6,nd_temp[pol],'k.',label='Model: EMSS')
        plt.grid()
        plt.legend()
    return fig  

def plot_ts(h5):
    ts = h5.timestamps - h5.timestamps[0]
    fig = plt.figure(figsize=(20,5))
    ant = h5.ants[0].name
    
    amp = [h5.vis[:,:,i] for i,p in enumerate(h5.corr_products) if (p[0]==p[1]) and (ant in p[0])] # Autocorr
    amp = np.sum(np.abs(amp), axis=0) # Total power
    plt.fill_between(ts, np.min(amp,axis=1), np.max(amp,axis=1), step="mid", facecolors='k', alpha=0.2)
    amp = np.mean(amp,axis=1)
    plt.plot(ts, amp, '.', label='Average of the data')
    
    on = h5.sensor['Antennas/'+ant+'/nd_coupler']
    plt.plot(ts, np.array(on).astype(float)*np.max(amp), 'g', label='katdal ND sensor')
    
    plt.legend()
    plt.title("Timeseries for antenna %s "%(ant))
    return fig


def read_and_plot_data(h5, ants=None, target='off1', select='track', Tbg=0,Toff=0, rfi_mask=None,
                       error_bars=False, pdf_output_dir=None):
    """ Load the data from the dataset and derive the system noise and noise diode temperatures.
        
        @param h5: an open katdal.Dataset
        @param Tbg & Toff: known values, as beam averages including Tcmb, Tbg with moon region blanked.
               C*Tcmb gets subtracted from Tbg if Tbg is nonzero.
        @param rfi_mask: boolean flags to match dataset.freqs (default None)
        @return: (freq [Hz], Tsys [K], Tsys_eta [K], Tnd [K])  NB: temperatures are ordered as (freq, [V,H])!
    """
    f_c = h5.spectral_windows[0].centre_freq
    Ku = (f_c > 11e9)
    
    out_Tsys, out_Tsys_eta, out_Tnd = [], [], []
    
    file_base = h5.name.split('/')[-1].split('.')[0]
    nice_filename =  file_base + '_T_sys_T_nd'

    h5.select(reset="F")
    if (rfi_mask is not None):
        h5.select(channels=~rfi_mask)
    freq = h5.channel_freqs
    
    ants = h5.ants if (ants is None) else [a for a in h5.ants if a.name in ants]
    
    observer = ants[0].observer
    observer.date = time.gmtime(h5.timestamps.mean())[:6] # katdal should NOT set this date to now()!
    
    pols = ['v','h']
    for ant in ants:
        ant = ant.name
        band = models.band(freq/1e6, ant=ant)
        ap_eff = models.Ap_Eff(band=band)
        try:
            rx_sn = h5.receivers[ant]
        except KeyError:
            logger.error('Receiver serial number for antennna %s not found in the H5 file'%ant)
            rx_sn = band.split("_")[-1].lower()+'.SN_NOT_FOUND'
        
        if pdf_output_dir:
            pdf_filename = pdf_output_dir+'/'+nice_filename+'.'+rx_sn+'.'+ant+'.pdf'
            pp = PdfPages(pdf_filename)
            
        h5.select(reset="T", ants=ant, targets=['Moon']+list(target))
        fig0 = plot_ts(h5)
        
        air_temp = h5.temperature.mean()
        all_nd_temp, all_TAc, all_eta_A = [], [], []
        for pol in pols:
            if not Ku:
                diode_filename = nd_models_folder + '/rx.'+h5.receivers[ant]+'.'+pol+'.csv'
                nd = scape.gaincal.NoiseDiodeModel(diode_filename)

            #cold data
            h5.select(ants=ant,pol=pol, targets=target,scans=select)
            
            cold_data = h5.vis[:].real
            cold_on,cold_off = get_nd_on_off(h5,log=logger)
            el_cold = h5.el.mean()*np.pi/180.
            #hot data
            h5.select(ants=ant,pol=pol, targets='Moon',scans=select)
            hot_on,hot_off = get_nd_on_off(h5,log=logger)
            hot_data = h5.vis[:].real
            el_hot = h5.el.mean()*np.pi/180.
            
            # APH changed these from mean -- more robust against RFI and also ND transitions
            cold_spec = np.median(cold_data[cold_off,:,0],0)
            hot_spec = np.median(hot_data[hot_off,:,0],0)
            cold_nd_spec = np.median(cold_data[cold_on,:,0],0)
            hot_nd_spec = np.median(hot_data[hot_on,:,0],0)

            if not Ku:
                nd_temp = nd.temperature(freq / 1e6)
                # antenna temperature on the moon (from diode calibration)
                TAh = hot_spec/(hot_nd_spec - hot_spec) * nd_temp
                # antenna temperature on cold sky (from diode calibration) (Tsys)
                TAc = cold_spec/(cold_nd_spec - cold_spec) * nd_temp
                logger.debug("Mean TAh = %f  mean TAc = %f "%(TAh.mean(),TAc.mean()))
            else:
                TAh = TAc = 0*freq
            
            Y = hot_spec / cold_spec
            R = 0.5*models.Dmoon(observer) # radius of the moon disc
            Os = 2*np.pi*(1-np.cos(R)) # APH changed from original np.pi * R**2 # disk source solid angle 
            fwhm = lambda f_Hz: 1.22*(models._c_/f_Hz)/ap_eff.D 
            HPBW = fwhm(freq)
            Om = 1.133 * HPBW**2  # main beam solid angle for a gaussian beam
            if Ku:
                eta_A = 0.7
            else:
                eta_A = ap_eff[pol.upper()*2](freq/1e6)
                #Ag = np.pi* (D/2.)**2 # Antenna geometric area
                #Ae = eta_A * Ag  # Effective aperture
                #lam = 299792458./freq # APH changed from 3e8
                #Om = lam**2/Ae # APH DON't replace gaussian approximation 4*log(2)/pi, which is within +/- 5% but systematically off -- K requires Om to use  gaussian approximation?
            x = 2*R/HPBW # ratio of source to beam
            K = ((x/1.2)**2) / (1-np.exp(-((x/1.2)**2))) # correction factor for disk source from Baars 1973
            # APH Note:  K == iint{psi(theta)}/iint{psi(theta)*U(theta)}  with psi(theta) = a pill box and assuming R < HPBW
            #        So  TA_src == iint{Tsrc(theta)*U(theta)}/iint{U(theta)} = iint{Tsrc_peak*psi(theta)*U(theta)}/iint{U(theta)}
            #                                                                = Tsrc_peak*1/K * iint{psi(theta)}/iint{U(theta)}
            #        It follows that  iint{psi(theta)} = Os  and  iint{U(theta)} = lambda^2/Ae
            #TA_moon = 225 * (Os/Om) * (1/K) # contribution from the moon (disk of constant brightness temp)
            TA_moon = models.Tmoon(freq/1e6,observer.date,HPBW) * (Os/Om) * (1/K) # APH changed from "255*(1/K)" to rather explicitly calculate beam-averaged temperature
            gamma = 1.0
            if False: # APH added Thot & Tcold and changed Tsys to Thot-Tcold instead of original TA_moon
                Thot = TA_moon
                Tcold = 0
            else: # TODO: merge to diodelib! APH proposed model, the background and atmosphere needs to be included in case elevation differs!
                C = (Os/K) / Om # Fraction of beam solid angle that is blocked by the Moon = iint_(theta<=R){U(theta)}/iint{U(theta)} = iint_{psi(theta)*U(theta)}/iint{U(theta)}
                logger.info("Fraction of beam that is NOT blocked by the moon: %.3f..%.3f"%(1-np.max(C),1-np.min(C)))
                Ton = TA_moon + ((Tbg-C*models.Tcmb)  if np.any(Tbg>0)  else 0) # At the top of atmosphere, Tbg is either 0 or is beam average with moon area blanked out but + 100% Tcmb
                tau = models.calc_atmospheric_opacity(h5.temperature.mean(), h5.humidity.mean()/100., h5.pressure.mean(), 1.1, freq/1e9) # APH added
                Tatm = 266.5 + 0.72*air_temp # APH 03/2017 as per Han & Westwater 2000
                # NB: the reference plane is at ideal antenna aperture -- excl. eta_rad!
                Thot = Ton*np.exp(-tau/np.sin(el_hot)) + Tatm*(1-np.exp(-tau/np.sin(el_hot)))
                Tcold = Toff*np.exp(-tau/np.sin(el_cold)) + Tatm*(1-np.exp(-tau/np.sin(el_cold)))
            
            Tsys = gamma * (Thot-Tcold)/(Y-gamma) # Tsys from y-method ... compare with diode TAc
            if error_bars:
                cold_spec_std = np.std(cold_data[cold_off,:,0],0)
                hot_spec_std = np.std(hot_data[hot_off,:,0],0)
                cold_nd_spec_std = np.std(cold_data[cold_on,:,0],0)
                hot_nd_spec_std = np.std(hot_data[hot_on,:,0],0)
                Y_std = Y * np.sqrt((hot_spec_std/hot_spec)**2 + (cold_spec_std/cold_spec)**2)
                Thot_std = 2.25
                gamma_std = 0.01
                Tsys_std = Tsys * np.sqrt((Thot_std/Thot)**2 + (Y_std/Y)**2 + (gamma_std/gamma)**2)
            else:
                Tsys_std = None
            
            if not Ku:
                Ydiode = hot_nd_spec / hot_spec
                #Tdiode = (TA_moon + Tsys)*(Ydiode/gamma-1)
                Tdiode_h = (Tsys-Tcold+Thot)*(Ydiode/gamma-1) # APH corrected from above -- Tsys as computed above includes Tcold
                # APH alternative to above, final result is the average which uses all available data (optimally)
                Ydiode = cold_nd_spec / cold_spec
                Tdiode_c = Tsys*(Ydiode/gamma-1)
                Tdiode = (Tdiode_h+Tdiode_c)/2.
                ## APH comment: alternative in SET_TR_005 eq 19 & 20 (calculate Tsys with ND on, take difference to get Tnd) - least sensitive to detector linearity
                #Ydiode = hot_nd_spec / cold_nd_spec
                #Tsys_nd = gamma * (Thot-Tcold)/(Ydiode-gamma)
                #Tdiode = Tsys_nd - Tsys
                ##Tdiode = (gamma*(Tsys+Thot-Tatm*(1-np.exp(-tau/np.sin(el_hot))))-Ydiode*(Tsys+Tcold-Tatm*(1-np.exp(-tau/np.sin(el_cold)))))/(Ydiode-gamma)
            
            if not Ku:
                out_Tnd.append(Tdiode); all_nd_temp.append(nd_temp)
            out_Tsys.append(Tsys)
            out_Tsys_eta.append(Tsys/eta_A)
            all_TAc.append(TAc)
            all_eta_A.append(eta_A)
        
        if (len(out_Tnd) > 0):
            Tdiode = {'v':out_Tnd[0],'h':out_Tnd[1]}
            nd_temp = {'v':all_nd_temp[0],'h':all_nd_temp[1]}
            fig2 = plot_nd(freq,Tdiode,nd_temp, ant=ant, file_base=file_base)
        Tsys = {'v':out_Tsys[0],'h':out_Tsys[1]}
        eta_A = {'v':all_eta_A[0],'h':all_eta_A[1]}
        TAc = {'v':all_TAc[0],'h':all_TAc[1]}
        Tsys_std = {'v':Tsys_std,'h':Tsys_std}
        fig1 = plot_Tsys_eta_A(freq,Tsys,eta_A,TAc, Tsys_std=Tsys_std, ant=ant, file_base=file_base, Ku=Ku)
        
        if pdf_output_dir:
            fig0.savefig(pp,format='pdf')
            fig1.savefig(pp,format='pdf')
            if (len(out_Tnd) > 0):
                fig2.savefig(pp,format='pdf')
            pp.close() # close the pdf file

    return freq, out_Tsys, out_Tsys_eta, out_Tnd


def process(dataset, ants, rfi_mask='../catalogues/rfi_mask.txt', freq_range=None, pdf_output_dir=None, doublecheck=False):
    """ Processes a dataset, potentially generating a T_noisediode output file as well as a PDF report.
        Use like:
            process(util.open_dataset(filename), ants=['m017'], rfi_mask='../catalogues/rfi_mask.txt')
        
        @param freq_range: select a subset to process, like (f_start, f_stop) [Hz] (default None).
        @return (freq [Hz], Tsys [K], Tsys_eta [K], Tnd [K])
    """
    if isinstance(dataset, str):
        dataset = util.open_dataset(dataset)
    filename = dataset.name.split()[0].split(" ")[0] # "rdb files are different
    
    # Set up freq_m
    dataset.select(reset="F")
    if freq_range:
        dataset.select(freqrange=freq_range)
    if rfi_mask:
        rfi_mask = util.load_rfi_static_mask(rfi_mask, dataset.freqs)
        dataset.select(reset="", channels=~rfi_mask)
    freq_m = dataset.channel_freqs
    
    # Print some diagnostics
    observer = dataset.ants[0].observer
    observer.date = ephem.Date(time.gmtime(dataset.timestamps.mean())[:6]) # katdal should NOT set this date to now()!
    logger.info("Great circle distance to the BIG 5 as observed from %s:" % str(observer))
    bright_targets = [("Sun", ephem.Sun(observer)), ("Moon", ephem.Moon(observer)), ("Jupiter", ephem.Jupiter(observer)), ("Saturn", ephem.Saturn(observer)), ("Venus", ephem.Venus(observer))]
    for tgt in dataset.catalogue.targets:
        tgt = tgt.body; tgt.compute(observer)
        logger.info("  For target %s at Astrometric RADEC=%s,%s"%(tgt,tgt.a_ra,tgt.a_dec))
        for name,xxx in bright_targets: # Below, "a_" i.e. Astrometric coordinates are required to correctly match planetary bodies to the sky map
            if (xxx.alt > 0):
                logger.info("\t%s: %.2f deg"% (name,ephem.separation((tgt.a_ra,tgt.a_dec),(xxx.a_ra,xxx.a_dec))*180/np.pi))
    
    # Process the data
    radec_2deg = lambda radec_rad: (np.asarray(radec_rad)*180/np.pi).tolist()
    Tbg = models.Tcmb + models.fit_gsm(*radec_2deg(dataset.catalogue['Moon'].radec(dataset.timestamps.mean())),
                                       radius_blank=0.5*models.Dmoon(time.gmtime(dataset.timestamps.mean())[:6]), debug=True)(freq_m)
    Toff1 = models.Tcmb + models.fit_gsm(*radec_2deg(dataset.catalogue['off1'].radec(dataset.timestamps.mean())), debug=True)(freq_m)
    Toff2 = models.Tcmb + models.fit_gsm(*radec_2deg(dataset.catalogue['off2'].radec(dataset.timestamps.mean())), debug=True)(freq_m)
    # Average over both 'off' measurements
    freq_m, Tsys_m, Tsys_eta_m, Tnd_m = read_and_plot_data(dataset, ants=ants, target=['off1','off2'],Tbg=Tbg,Toff=(Toff1+Toff2)/2.,
                                                 rfi_mask=rfi_mask, pdf_output_dir=pdf_output_dir)
    
    # This gives equivalent results "to within the noise"
    if doublecheck: 
        freq_m1, Tsys_m1, Tsys_eta_m1, Tnd_m1 = read_and_plot_data(dataset, ants=ants, target='off1',Tbg=Tbg,Toff=Toff1, rfi_mask=rfi_mask)
        freq_m2, Tsys_m2, Tsys_eta_m2, Tnd_m2 = read_and_plot_data(dataset, ants=ants, target='off2',Tbg=Tbg,Toff=Toff2, rfi_mask=rfi_mask)
        Tsys_m = np.mean(np.stack([Tsys_m1,Tsys_m2],-1), axis=-1)
        Tsys_eta_m = np.mean(np.stack([Tsys_eta_m1,Tsys_eta_m2],-1), axis=-1)
        Tnd_m = np.mean(np.stack([Tnd_m1,Tnd_m2],-1), axis=-1)
        # Summarise the difference between ND calibration results
        if (len(Tnd_m1) > 0) and (len(Tnd_m2) > 0):
            plt.figure(figsize=(16,4))
            plt.title(filename)
            plt.plot(freq_m/1e6, np.asarray(Tnd_m1[0])-np.asarray(Tnd_m2[0]), label="V")
            plt.plot(freq_m/1e6, np.asarray(Tnd_m1[1])-np.asarray(Tnd_m2[1]), label="H")
            plt.xlabel("f [MHz]"); plt.ylabel("Tnd_1-Tnd_2 [K]"); plt.legend()
            plt.grid(True); plt.ylim(-5,5)
            if freq_range:
                plt.xlim(*[_/1e6 for _ in freq_range])

    return (freq_m, Tsys_m, Tsys_eta_m, Tnd_m)


def write_Tnd(dataset, freq, TndVH, ants=None, output_dir=None, smooth=5, fignum=-1, style='m-'):
    """ Create a standard calibration file for the noise diode temperature data given.
        @param freq: [Hz]
        @param TndVH: [V_ant0,H_ant0, ... V_antn,H_antn]
        @param ants: the names of the antennas for which the data is given or None if all from the dataset (default None).
        @param output_dir: specify this to override `nd_models_folder` (default None).
        @param smooth: in multiples of channels, must be odd.
        @param fignum: figure number for debug plots; a negative number suppresses this figure.
    """
    if isinstance(dataset, str):
        dataset = util.open_dataset(dataset)
    filename = dataset.name.split()[0].split(" ")[0] # "rdb files are different
    output_dir = output_dir if (output_dir) else nd_models_folder
    
    ants = dataset.ants if (ants is None) else [a for a in dataset.ants if a.name in ants]
    fig = plt.figure(fignum) if not (fignum is None or fignum >= 0) else None
    for i,ant in enumerate(ants):
        for pol,Tdiode in enumerate(TndVH[i*2:(i+1)*2]):
            _f, _T = freq[smooth//2::smooth], sp.signal.medfilt(Tdiode,smooth)[smooth//2::smooth]
            if (fig is not None):
                plt.subplot(1,2,pol+1); plt.plot(_f/1e6,_T,style, label=ant.name)
            rx_band,rx_SN = dataset.receivers[ant.name].split('.')
            outfile = open('%s/rx.%s.%s.%s.csv' % (output_dir, rx_band.lower(), rx_SN, "vh"[pol]), 'w')
            outfile.write('# Tsys hot-cold result based on %s\n# Frequency [Hz], Temperature [K]\n'%filename)
            # Write CSV part of file
            if (rx_band.upper() == "U"): # unflip the flipped first nyquist ordering & skip low freq garbage
                outfile.write(''.join(['%s, %s\n' % (entry[0], entry[1]) for entry in zip(np.flipud(_f[_f>540e6]),np.flipud(_T[_f>540e6]))]))
            else:
                outfile.write(''.join(['%s, %s\n' % (entry[0], entry[1]) for entry in zip(_f,_T)]))
            outfile.close()
            logger.debug("Successfuly created file %s" % outfile.name)
    if (fig is not None):
        plt.legend()


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser(usage="%prog [opts] <dataset filename>",
                                   description="Process a 'Tsys on the Moon' dataset and generate a standard report.")
    parser.add_option("-a", "--ants", default="", help="The name(s) of the specific antenna(s) in the dataset to process, or 'all'")
    parser.add_option("--freq-mask", default="", help="Filename for frequency mask, either text or pickle format.")
    parser.add_option("--select-freq", default="", help="Comma-separated start & stop frequencies in Hz, e.g. '900e6,1600e6'")
    parser.add_option("-s", "--save-nd", action="store_true", help="Generate standard noise diode calibration files")
    opts, args = parser.parse_args()
    
    filename = args[0]
    select_freq = None if not opts.select_freq else map(float, opts.select_freq.split(","))
    freq, Tsys, Tsys_eta, Tnd = process(filename, ants=opts.ants, rfi_mask=opts.freq_mask, freq_range=select_freq, pdf_output_dir="./")
    if opts.save_nd:
        write_Tnd(filename, freq, Tnd, opts.ants)
    