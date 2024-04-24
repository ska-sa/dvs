"""
    Top-level package of analysis routines for the Dish Verification System.
"""
import analyze_fastgain as fastgain
import analyze_SEFD as driftscan
import analyze_tipping as tipcurve

cbid2url = lambda cbid: "http://archive-gw-1.kat.ac.za/%s/%s_sdp_l0.full.rdb"%(cbid,cbid) # Only works from inside the SARAO firewall
