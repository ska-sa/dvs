aperture_efficiency
-------------------
MK_L & MK_U: simply copied & renamed from https://katfs.kat.ac.za/svnSystemEngineering/MeerKAT/4%20System%20Level5/5%20Integration%20&%20Verification/Analysis/Receptor/Predictions/SpilloverAndEfficiency/ant_eff_*_AsBuilt.csv

SKA_B1 & SKA_B2: converted from SKADCB12_EMResults-20190808.zip:Comp_ang__Ae.png to .csv by using e.g. WebPlotDigitiser, then open as spreadsheet and divide by PI*(15.3/2)^2 to convert from Ae to ant_eff (see SKADC\Qualification\src\photocloud.py).
Format to 1 decimal point and add the following 2 lines as header
    # Digitised from SKADCB12_EMResults-20190808.zip:Comp_ang_B1_Ae.png by R. Lehmensiek 08/08/2019 and normalized to represent D_g=15.3 m.
    # f [MHz]	eta_ap [%]



beam_patterns
-------------
mkat: simply copied from https://katfs.kat.ac.za/svnSystemEngineering/MeerKAT/4%20System%20Level5/5%20Integration%20&%20Verification/Analysis/Receptor/Predictions/BeamPatterns

ska: simply extracted from EMSS 2019_08_06_SKA_45_B1.zip  2019_08_06_SKA_45_B2.zip  2019_08_06_SKA_Ku.zip



spill-over
----------
MK_L & MK_U: simply copied from https://katfs.kat.ac.za/svnSystemEngineering/MeerKAT/4%20System%20Level5/5%20Integration%20&%20Verification/Analysis/Receptor/Predictions/SpilloverAndEfficiency/ant_eff_*_AsBuilt.csv
(converted from EMSS .mat files using (https://katfs....)/convertmat.py)

SKA_B1 & SKA_B2: extracted from EMSS 2019_10_02_Ta.zip & converted Ta___.mat to .dat by using https://svn.atnf.csiro.au/skadc/L4%20Dish%20Element%20System%20Engineering/Dishes%20Element%20Product%20Data/Qualification/src/sensitivity.py
>> 2019_10_02_Ta.zip, received from iptheron@emss.co.za on 11 October 2019
>>    "Ta_{15,45,90}_B2.mat" bevat antenna temperature {Ta, Ta20} oor frekwensies en zenith hoeke vir elkeen van die "as built" skottels. Ta(freq,th_p,pol), waar pol = 1 vertikaal en pol = 2 horisontaal is, gee die totale "tipping curve noise temperature". Ta20(freq,th_p,pol) verteenwoordig dieselfde maar net van die hoefbundel tot -20dB.
>>    Let op dat die tipping curves eintlik net streng fisies korrek is by die elevasie hoek wat deur die "as built" model verteenwoordig word.


noise-diode-models
------------------
Noise diode models that are "work in progress" (not yet committed to https://github.com/ska-sa/katconfig)

receiver-models
---------------
Receiver noise models that are "work in progress" (not yet committed to https://github.com/ska-sa/katconfig)
