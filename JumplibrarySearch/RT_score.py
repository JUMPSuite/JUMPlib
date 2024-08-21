import pandas as pd
import pyteomics
from pyteomics import mzxml
import numpy as np
import scipy
from scipy import stats
from scipy import interpolate
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
from logFunctions import *

from pyteomics import mass
from statsmodels.distributions.empirical_distribution import ECDF
from elutionCases import *
from RTfunctions import *
from os.path import dirname


def psms_after_search_rt_extract(psms_Rt):
    required_columns = psms_Rt.columns

    quantiles = []
    for val in range(1,21):
    #     quantiles.append("q{}".format(val))
        quantiles.append(val)
    try:
#        psms_Rt['JDscore_q_bins'] = pd.qcut(psms_Rt['JDscore'], 21, quantiles, duplicates='drop')
        psms_Rt['JDscore_q_bins'] = pd.qcut(psms_Rt['JDscore'], 21, duplicates='drop')
    except:
        print (" .... Issue with binning. Trying different bin size")
        try:
#            psms_Rt['JDscore_q_bins'] = pd.qcut(psms_Rt['JDscore'], 20, quantiles, duplicates='drop')
            psms_Rt['JDscore_q_bins'] = pd.qcut(psms_Rt['JDscore'], 20, duplicates='drop')            
        except:
            print (" .... Issue with binning.")

    quantiles_PSMS = psms_Rt[["keys","JDscore_q_bins"]].groupby("keys", as_index=False).agg(max)
    quantiles_PSMS["psms_quantile_keys"] = quantiles_PSMS["keys"]+"___"+quantiles_PSMS["JDscore_q_bins"].astype("str")
    psms_Rt['psms_quantile_keys'] = psms_Rt["keys"]+"___"+psms_Rt["JDscore_q_bins"].astype("str")

    extractRT_PSMS = psms_Rt[psms_Rt['psms_quantile_keys'].isin(quantiles_PSMS["psms_quantile_keys"])]
    return extractRT_PSMS[required_columns]


def rt_score(searchFile, mzxml, outputFolder, logFile):
    try:
        psms = pd.read_csv(searchFile)
    except:
        print ("Could not find csv file as search result. Trying excel file")
        try:
            psms = pd.read_excel(searchFile)
        except:
            print ("Input search file format is incorrect. Please provide correct format")

    psms["z"]=psms.Outfile.str.split(".", expand =True)[3]
    psms["keys"] = psms.Peptide+"__"+psms.z.astype("str")
    runName = os.path.basename(mzxml).split(".")[0]


    # get the quantile based psms for best scoring psms
    extractRT_PSMS = psms_after_search_rt_extract(psms)

    ext_data_dict,res=inferRT_afterSearch(extractRT_PSMS, [mzxml], eps=1)
    psms_Rt = psms.merge(res, how="inner",left_on="keys", right_on = "key")

    print ("Generating Loess curve to allign exp RT with library RT\n")
    print ("Criteria used for creating the Loess curve\n...1) JDscore > 0.95\n...2) Only Target peptides considered\n...3) Met oxidized peptides removed from modeling")	

    deltaRT_recorder = alignRT_aftersearch(psms_Rt, runName, tol_min=1)


    #delta RT is library RT - infered RT
    psms_Rt["deltaRT_postcal"] = psms_Rt.RT.astype("float") - psms_Rt.calibratedRTs.astype("float")

    # residuals = np.array(Y)-np.array(X)
    residuals = np.array(psms_Rt["deltaRT_postcal"] )

    ecdfRt = ECDF(abs(residuals))
    # pRt = ecdfRt(abs(rtShift))

    psms_Rt["pRT"] = psms_Rt.apply(lambda x: ecdfRt(abs(x.deltaRT_postcal)), axis=1)
    psms_Rt["pRT"] = psms_Rt.apply(lambda x: max(np.finfo(float).eps, x.pRT), axis=1)

    psms_Rt["pMs2"] = 1 - psms_Rt.JDscore  # p-value-like score (the smaller, the better) --- simMs2 is JDscore
    psms_Rt["pMs2"] = psms_Rt.apply(lambda x: max(np.finfo(float).eps, x.pMs2), axis=1)


    psms_Rt["combinePval"] = psms_Rt.apply(lambda x: combine_p_values(x.pMs2, x.pRT), axis=1)
    psms_Rt["combined_score"]= psms_Rt.apply(lambda x: abs(-np.log10(x.combinePval)), axis=1)

    # all_columns +=["pRT","combined_score"]

    psms_Rt.to_csv(outputFolder+"/"+outputFolder+".allScores.csv",index=None)


    # psms_Rt[all_columns].to_csv(outputFolder+"/"+outputFolder+".combinedScore.csv",index=None)




def combine_p_values(pMs2, pRt):
    p = 1 - stats.chi2.cdf(-2 * (np.log(pMs2) + np.log(pRt)), 4)    # Fisher's method
    return p


def genLowessFunction(df, minv1, maxv1,minv2, maxv2):
    X=[min(minv1,minv2)]+list(df.RT)+[max(maxv1,maxv1)]
    miny = min(list(df.peptide_RT))
    maxy = max(list(df.peptide_RT))
    Y=[minv2]+list(df.peptide_RT)+[maxv2]
    
    lowess = sm.nonparametric.lowess
    #Functions: lowess Fit a smooth nonparametric regression curve to a scatterplot
    Z = lowess(Y,X,frac=1/3,it=3, return_sorted = False)
    #predicted y values using Z function
    y_ = interpolate.interp1d(X, Z, bounds_error=False)
    
    ### to generate the lowess curve use the xnew and ynew 
    #making a numpy array ranging from smallest reference/new library RT to largest
    #compute minimum and maximum value for the curve generation
#     minv = np.min(list(dfR_T.RT)+list(dfL_T.RT))
#     maxv = np.max(list(dfR_T.RT)+list(dfL_T.RT))
    xnew = np.arange(miny, maxy, 0.0001)
    
#     predicting ynew with the function y_
    ynew = y_(xnew)
    
    return y_, xnew, ynew, Y,X


def combine_p_val(expsheet):
    p_values_combo = []
    for rownum, rowdata in expsheet.iterrows():
        tempy = [rowdata.RT_score,rowdata.P_val_like_JDScore]
        comboZscore, combopvalue = stats.combine_pvalues(pvalues=tempy,method='stouffer')
        p_values_combo.append(combopvalue)
    expsheet["combined_p_values"] = p_values_combo



