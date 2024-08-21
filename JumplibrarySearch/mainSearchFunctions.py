import pandas as pd 
import re
import numpy as np
import math

from normalization_PSMSHandler import *
import scipy
import scipy.stats
from otherScores import * 
import time
from logFunctions import *

pd.options.mode.chained_assignment = None  # default='warn'
from spectra_process import * # get_spec_df_from_pkl, get_spec_df_from_ms2, entropy_sim_per_peak, get_similarity


#this function trims the raw spectrum by comparing with spectral Library within the given tolerance
#this is different than Ji-Hoon's metabolomics library because all the ions in spectral library are fixed 
#the raw spectrum ions are adjusted based on spectral library

#this function prepares the input for Ji-Hoon's program such that the top intensity sorted ions will be mostly 
#matched to the spectral library

#this is key as for Ji-Hoon's spectral library also has noise so he sorts the library based on intensity to get the 
#most intense ions. Here we first trim the spectra and later get the 

#Trying to reduce computational time by trimming the m/z if it is out of tolerance withouth going to SD determining step

#this function trims the raw spectrum by comparing with spectral Library within the given tolerance
#this is different than Ji-Hoon's metabolomics library because all the ions in spectral library are fixed 
#the raw spectrum ions are adjusted based on spectral library

#this function prepares the input for Ji-Hoon's program such that the top intensity sorted ions will be mostly 
#matched to the spectral library

#this is key as for Ji-Hoon's spectral library also has noise so he sorts the library based on intensity to get the 
#most intense ions. Here we first trim the spectra and later get the 

#Trying to reduce computational time by trimming the m/z if it is out of tolerance withouth going to SD determining step

#if multiple produc ions matches one product ions the intensity will be summed together for dot product calculation

def trimFeatSpec(featSpec,tolInputDict,int_sd_dict,tolerance_type,libSpec, ms2_tol):
    #values for dynamic tolerance search

    if tolerance_type.upper() == "DYNAMIC":
        maximumInt = max(int_sd_dict.keys())
        minimumInt = min(int_sd_dict.keys())
        checkToleranceLevel = int_sd_dict[minimumInt]

    tol_max=ms2_tol #can be the paramter
    #trim the spectra in featSpec to make it similar to library to fit in Ji-Hoon's code

    checkMaxLibMZ = np.max(libSpec["mz"])
    checkMinLibMZ = np.min(libSpec["mz"])

    #to avoid unnecessary tolerance calculation directly use the maximum and min ions in library 
    #with some tolerance to cutoff unwanted spectra
    max_exp_mzCheck = float(checkMaxLibMZ)+(checkMaxLibMZ/1000000*tol_max)
    min_exp_mzCheck = float(checkMinLibMZ)-(checkMinLibMZ/1000000*tol_max)

    exp_mz_list = featSpec["mz"]
    intensity_list = featSpec["intensity"]
    
    if tolerance_type.upper() == "DYNAMIC":
        log10_intensity_list = tolInputDict["intensity"]

    calibrationList = []
    calibrationListInt = []
    calibrationListIntTolerance = []
    for i, val in enumerate(exp_mz_list):
        if (val <= max_exp_mzCheck) and (val >= min_exp_mzCheck):
            calibrationList.append(val)
            calibrationListInt.append(intensity_list[i])

            if tolerance_type.upper() == "DYNAMIC":
                calibrationListIntTolerance.append(log10_intensity_list[i])

    #make a trimeed spectral dictionary and update it as the ion is within the tolerance
    tr_featSpec = {}
    
    
    # matchedLib = []
    #     print (calibrationListIntTolerance)
    for libIndex, mzLib in enumerate(libSpec["mz"]):
        for index, masses in enumerate(calibrationList):
            massshift = ppmCalc(float(mzLib), float(masses), tol=tol_max)
            if tolerance_type.upper() == "DYNAMIC":
                if abs(massshift) > checkToleranceLevel:
                    pass
                else:

                    #np.round(dfDyn.log10Intensity,5)
                    checkTolIntensity = np.round(float(calibrationListIntTolerance[index]),5)
        #                 print (checkTolIntensity)
                    if checkTolIntensity > maximumInt: #for the intensity not present in dictionary and it is maximum
                        checkTolIntensity = maximumInt
                    if checkTolIntensity < minimumInt: #for the intensity not present in dictionary and it is minimum
                        checkTolIntensity = minimumInt

                    tol_max = int_sd_dict[checkTolIntensity]
            else:
                
                if abs(massshift) <= tol_max:
        #                 print (massshift)
                    if mzLib not in tr_featSpec.keys():
                        tr_featSpec[mzLib]=[calibrationListInt[index]]
                    else:
                        tr_featSpec[mzLib].append(calibrationListInt[index])
                    
                    # matchedLib.append(mzLib)


    tr_featSpec2 = {"mz":[],"intensity":[]}

    for ions in libSpec["mz"]:
        #add intensity 0 to all the ions that are present in library but are not within the given tolerance in spectrum raw
        if ions in tr_featSpec.keys():
            tr_featSpec2["mz"].append(ions)
            tr_featSpec2["intensity"].append(np.max(tr_featSpec[ions]))
        else:
            tr_featSpec2["mz"].append(ions)
            tr_featSpec2["intensity"].append(0.0)

    return tr_featSpec2



#this fucntion is necessary to remove the product ions in library that is within the tolerance level. Sometimes differente theoretical ions are indistinguishable for example see example file https://github.com/surPoudel/JUMPp-lib

#library entry p0045203
def cleanLibRedundancy(libSpec, ms2_tol):
    cleanLib = {"mz":[],"intensity":[]}
    mzArray = sorted(libSpec["mz"])
    mzDict = {}
    val = 0
    for mz in mzArray:
        if abs(ppmCalc(mz, val)) <= ms2_tol:
            mzDict[mz] = val
        else:
            mzDict[mz] = mz
            val = mz
    #unique keys 
    
    keys = mzDict.values()
    
    for value in sorted(set(keys)):
        index = libSpec["mz"].index(value)
        cleanLib["mz"].append(value)
        cleanLib["intensity"].append(libSpec["intensity"][index])
    return cleanLib

#add a function to select top N product ions for rough JDscore computation

def quickScoreLibClean(libSpec, ms2_tol):
    #ind = np.argsort([-i for i in [1,2,3,4,5]])
    cleanLib = {"mz":[],"intensity":[]}
    mzArray = sorted(libSpec["mz"])
    mzDict = {}
    val = 0
    for mz in mzArray:
        if abs(ppmCalc(mz, val)) <= ms2_tol:
            mzDict[mz] = val
        else:
            mzDict[mz] = mz
            val = mz
    #unique keys 
    
    keys = mzDict.values()
    
    for value in sorted(set(keys)):
        index = libSpec["mz"].index(value)
        cleanLib["mz"].append(value)
        cleanLib["intensity"].append(libSpec["intensity"][index])
    return cleanLib



def trimFeatSpecQC(featSpec,tolInputDict,int_sd_dict,tolerance_type,libSpec):
    #values for dynamic tolerance search

    if tolerance_type.upper() == "DYNAMIC":
        maximumInt = max(int_sd_dict.keys())
        minimumInt = min(int_sd_dict.keys())
        checkToleranceLevel = int_sd_dict[minimumInt]

    tol_max=10 #can be the paramter
    #trim the spectra in featSpec to make it similar to library to fit in Ji-Hoon's code

    checkMaxLibMZ = np.max(libSpec["mz"])
    checkMinLibMZ = np.min(libSpec["mz"])

    #to avoid unnecessary tolerance calculation directly use the maximum and min ions in library 
    #with some tolerance to cutoff unwanted spectra
    max_exp_mzCheck = float(checkMaxLibMZ)+(checkMaxLibMZ/1000000*tol_max)
    min_exp_mzCheck = float(checkMinLibMZ)-(checkMinLibMZ/1000000*tol_max)

    exp_mz_list = featSpec["mz"]
    intensity_list = featSpec["intensity"]

    if tolerance_type.upper() == "DYNAMIC":
        log10_intensity_list = tolInputDict["intensity"]

    calibrationList = []
    calibrationListInt = []
    calibrationListIntTolerance = []
    for i, val in enumerate(exp_mz_list):
        if (val <= max_exp_mzCheck) and (val >= min_exp_mzCheck):
            calibrationList.append(val)
            calibrationListInt.append(intensity_list[i])

            if tolerance_type.upper() == "DYNAMIC":
                calibrationListIntTolerance.append(log10_intensity_list[i])

    #make a trimeed spectral dictionary and update it as the ion is within the tolerance
    tr_featSpec = {"mz":[],"intensity":[]}

    matchedLib = []
    #     print (calibrationListIntTolerance)
    for mzLib in libSpec["mz"]:
        for index, masses in enumerate(calibrationList):
            massshift = ppmCalc(float(mzLib), float(masses), tol=tol_max)
            if tolerance_type.upper() == "DYNAMIC":
                if abs(massshift) > checkToleranceLevel:
                    pass
                else:

                    #np.round(dfDyn.log10Intensity,5)
                    checkTolIntensity = np.round(float(calibrationListIntTolerance[index]),5)
        #                 print (checkTolIntensity)
                    if checkTolIntensity > maximumInt: #for the intensity not present in dictionary and it is maximum
                        checkTolIntensity = maximumInt
                    if checkTolIntensity < minimumInt: #for the intensity not present in dictionary and it is minimum
                        checkTolIntensity = minimumInt

                    tol_max = int_sd_dict[checkTolIntensity]
            else:
                if abs(massshift) > tol_max:
                    pass
            if abs(massshift) < tol_max:
    #                 print (massshift)
                if masses not in tr_featSpec["mz"]:
                    tr_featSpec["mz"].append(masses)
                    tr_featSpec["intensity"].append(calibrationListInt[index])
                    matchedLib.append(mzLib)



    #check the unmatched ions and make a list of unmatched ions (if any). This should be very less but this is for exceptional case

    unmatchedLib = []
    for ions in libSpec["mz"]:
        if ions not in matchedLib:
            unmatchedLib.append(ions)


    #     #seek the closest ion that matches with the unmatched ions
    #     #extract the mz information from specLib and its intensity too
    #     #use exp_mz_list to generate closest ion for unmatched ions

    for unmatchedIons in unmatchedLib:
        #add intensity 0 to all the ions that are present in library but are not within the given tolerance in spectrum raw
        tr_featSpec["mz"].append(unmatchedIons)
        tr_featSpec["intensity"].append(0.0)

    return tr_featSpec, len(matchedLib)



def scanPrecursorMatch(precMZ, charge, libDF, tol):
    max_prec_mzCheck = precMZ+(precMZ/1000000*tol)
    min_prec_mzCheck = precMZ-(precMZ/1000000*tol)
    #print ("minimum prec mz = ", min_prec_mzCheck)
    #print ("maximum prec mz = ", max_prec_mzCheck) 
    #scanning for precursor match that is within the tolerance
    #df = df[df['closing_price'].between(99, 101)]

    # libDF_matched = libDF[libDF.charge == charge]
    # libDF_matched = libDF[libDF.prec_MZ.astype("float").between(min_prec_mzCheck, max_prec_mzCheck)]


    libDF_matched = libDF[(libDF.prec_MZ.astype("float").between(min_prec_mzCheck, max_prec_mzCheck)) & (libDF.charge == charge)]

    # libDF_matched = libDF[(libDF.prec_MZ.astype("float") >= min_prec_mzCheck) & (libDF.prec_MZ.astype("float") <= max_prec_mzCheck) & (libDF.charge == charge)]
    #print (libDF_matched)
    return libDF_matched

def calcMS2SimilaritySuresh(featSpec, libSpec):
    num = np.dot(np.sqrt(featSpec["intensity"]),np.sqrt(libSpec["intensity"]))
    return num


def calcMS2Similarity(featSpec, libSpec, tol):
    # Calculation of MS2 similarity between a feature and a library compound
    # Reference: Clustering millions of tandem mass spectra, J Proteome Res. 2008; 7: 113-22

    # Input arguments
    # featSpec (dictionary): MS2 spectrum of a feature (key = "mz", "intensity")
    # libSpec (dictionary): MS2 spectrum of a library compound (key = "mz", "intensity", "index" (ignorable))
#     nPeaks = int(params["num_peaks_ms2_similarity"])    # Default = 30 according to the above reference
    #nPeaks = 30
    #nPeaks = len(libSpec["mz"]) #we are using all ions in the library spectra
    #k = min(nPeaks, min(len(featSpec["mz"]), len(libSpec["mz"])))

    # Keep $k strongest peaks in both spectra
    # featDict[mz] = intensity
    # libDict[mz] = intensity
    featDict, libDict = {}, {}
    ind = np.argsort([-i for i in featSpec["intensity"]])
    for i in ind:
        featDict[featSpec["mz"][i]] = featSpec["intensity"][i]
    ind = np.argsort([-i for i in libSpec["intensity"]])
    for i in ind:
        libDict[libSpec["mz"][i]] = libSpec["intensity"][i]

    # Join two sets of m/z values and make a new set of unique m/z values
    # Duplicate masses are removed as follows
    # - We consider two peaks to have a similar mass if they are within 0.5 Da from each other)
    # - For those two peaks having similar mass, the lower one will be the unique one
    #   (e.g. One peak with m/z = 100 and the other peak with m/z = 100.4 -> they will be merged to m/z = 100)
    mzArray = list(featDict.keys()) + list(libDict.keys())
    mzArray = sorted(mzArray)
    mzDict = {}
    val = 0
    for mz in mzArray:
        if abs(ppmCalc(mz, val)) <= tol:
        #if abs(mz - val) <= 0.5:
            mzDict[mz] = val
        else:
            mzDict[mz] = mz
            val = mz

    # Reduction of spectrum to a vector by assigning to each intensity to the unique m/z bins
    # And then, calculate the similarity; normalized dot-product
    s = {}
    for key, val in mzDict.items():
        s[val] = {}
        s[val]["feat"] = 0
        s[val]["lib"] = 0

    for key, val in mzDict.items():
        if key in featDict:
            s[val]["feat"] += np.sqrt(featDict[key])
        if key in libDict:
            s[val]["lib"] += np.sqrt(libDict[key])
#    num = 0
#    for mz in s.keys():
#        num += s[mz]["feat"] * s[mz]["lib"]
    num, den1, den2 = 0, 0, 0    # numerator, denominator1, denominator2
    for mz in s.keys():
        num += s[mz]["feat"] * s[mz]["lib"]
        den1 += s[mz]["feat"] ** 2
        den2 += s[mz]["lib"] ** 2

    if den1 * den2 == 0:
        normDotProduct = 0
    else:
        normDotProduct = num / np.sqrt(den1 * den2)

    return normDotProduct
#    return num

#this is the main library search function that generates the score of matches




#result = librarySearchMain(x,libDF,top_ions_control, min_top_ions, top_ions,ms2_tol,ms1_tol,top_ions_per_bin,binsize,tolerance_type,int_sd_dict, method)

def librarySearchMain(expDF,libDF,top_ions_control, min_top_ions, top_ions,ms2_tol,ms1_tol,top_ions_per_bin,binsize,tolerance_type,int_sd_dict, method):
    # dfDict = {}#collection of matched precursor MZ dataframe
    # ms2_tol = 15 #this is parameter for precursor ion matches
    '''
    mz_cols = list(expDF.columns)
    np_arr = expDF.to_numpy()
    '''
    mz_cols1 = list(expDF.columns)
    np_arr1 = expDF.to_numpy()
    # top_ion=3
    dot_product_results = {}
    cnt = 0
    update = 10000
    unmatched_precursors = 0
    
    
    # initial params
    # method = 'normalized_dot_product' # PSM scoring: normalized_dot_product, entropy, hyperscore
    # split_n = 4
    norm_1e2 = 1e2 # RT
    norm_1e4 = 1e4 # prec_MZ
    # top_ions = 10
    # binsize = 100
    # ms1_tol = 10
    # ms2_tol = 10
    # factorial[0,1,2,3,4,5,...,101]
    # log10factorial[0,1,2,3,4,5,...,101]
    # factorial=[1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600,6227020800,87178291200,1307674368000,20922789888000,355687428096000,6.40237370572800e+15,1.21645100408832e+17,2.43290200817664e+18,5.10909421717094e+19,1.12400072777761e+21,2.58520167388850e+22,6.20448401733239e+23,1.55112100433310e+25,4.03291461126606e+26,1.08888694504184e+28,3.04888344611714e+29,8.84176199373970e+30,2.65252859812191e+32,8.22283865417792e+33,2.63130836933694e+35,8.68331761881189e+36,2.95232799039604e+38,1.03331479663861e+40,3.71993326789901e+41,1.37637530912263e+43,5.23022617466601e+44,2.03978820811974e+46,8.15915283247898e+47,3.34525266131638e+49,1.40500611775288e+51,6.04152630633738e+52,2.65827157478845e+54,1.19622220865480e+56,5.50262215981209e+57,2.58623241511168e+59,1.24139155925361e+61,6.08281864034268e+62,3.04140932017134e+64,1.55111875328738e+66,8.06581751709439e+67,4.27488328406003e+69,2.30843697339241e+71,1.26964033536583e+73,7.10998587804864e+74,4.05269195048772e+76,2.35056133128288e+78,1.38683118545690e+80,8.32098711274139e+81,5.07580213877225e+83,3.14699732603879e+85,1.98260831540444e+87,1.26886932185884e+89,8.24765059208247e+90,5.44344939077443e+92,3.64711109181887e+94,2.48003554243683e+96,1.71122452428141e+98,1.19785716699699e+100,8.50478588567862e+101,6.12344583768861e+103,4.47011546151268e+105,3.30788544151939e+107,2.48091408113954e+109,1.88549470166605e+111,1.45183092028286e+113,1.13242811782063e+115,8.94618213078297e+116,7.15694570462638e+118,5.79712602074737e+120,4.75364333701284e+122,3.94552396972066e+124,3.31424013456535e+126,2.81710411438055e+128,2.42270953836727e+130,2.10775729837953e+132,1.85482642257398e+134,1.65079551609085e+136,1.48571596448176e+138,1.35200152767840e+140,1.24384140546413e+142,1.15677250708164e+144,1.08736615665674e+146,1.03299784882391e+148,9.91677934870949e+149,9.61927596824821e+151,9.42689044888324e+153,9.33262154439441e+155,9.33262154439441e+157,9.42594775983835e+159]
    log10factorial=[0,0,0.301029995663981,0.778151250383644,1.38021124171161,2.07918124604762,2.85733249643127,3.70243053644553,4.60552052343747,5.55976303287679,6.55976303287679,7.60115571803502,8.68033696408264,9.79428031638948,10.9404083520677,12.1164996111234,13.3206195937793,14.5510685151576,15.8063410202609,17.0850946212137,18.3861246168777,19.7083439116116,21.0507665924338,22.4124944284514,23.7927056701630,25.1906456788351,26.6056190268059,28.0369827909649,29.4841408223071,30.9465388202061,32.4236600749257,33.9150217687600,35.4201717470799,36.9386856869578,38.4701646040000,40.0142326483503,41.5705351491176,43.1387368731846,44.7185204698014,46.3095850768279,47.9116450681559,49.5244289248756,51.1476782152735,52.7811466708531,54.4245993473393,56.0778118611146,57.7405696927962,59.4126675507319,61.0939087881075,62.7841048681360,64.4830748724720,66.1906450485700,67.9066483922048,69.6309242618056,71.3633180216285,73.1036807111228,74.8518687381290,76.6077435938015,78.3711715873644,80.1420235990066,81.9201748493902,83.7055046844010,85.4978963738992,87.2972369233528,89.1034168973367,90.9163302539795,92.7358741895214,94.5619489922222,96.3944579049285,98.2333069956657,100.078405035680,101.929663384399,103.786995880830,105.650318740951,107.519550460682,109.394611724073,111.275425316354,113.161916041527,115.054010644217,116.951637735508,118.854727722500,120.763212741378,122.677026593762,124.596104686138,126.520383972200,128.449802897914,130.384301349158,132.323820601776,134.268303273927,136.217693280571,138.171935790011,140.130977182332,142.094765009677,144.063247958231,146.036375811831,148.014099417120,149.996370650160,151.983142384426,153.974368460118,155.970003654716,157.970003654716,159.974325028498]
    score_cutoff = 0.8
    match_cutoff = 4
    coverage_cutoff = 0.4
    peplen_cutoff = 10
    match_threshold = 3
    
    candino=np.zeros((len(expDF["scan"]),2))# candino
    # 1. scan each exp_scan
    for tno in range(0,len(np_arr1)):
        row1=np_arr1[tno]
        cnt+=1
        
        # scanKey = str(scan)+"."+str(charge)+"."+str(precMZ)
        # load pkl (lib_data.columns: Index(['scan', 'charge', '[M+H]+', 'prec_MZ', 'L_ID', 'L_peptide', 'L_protein', 'RT', 'm/z', 'intensity', 'norm_factor']) )
        exp_scan = int( row1[mz_cols1.index("scan")] )
        exp_prech = int( row1[mz_cols1.index("charge")] )
        exp_prec_mz = float( row1[mz_cols1.index("prec_MZ")] )
        exp_normf = row1[mz_cols1.index("norm_factor")]
        
        exp_mz = np.array(row1[mz_cols1.index("m/z")])
        exp_inten = np.array(row1[mz_cols1.index("intensity")])
        exp_inten2 = exp_inten*exp_normf/(norm_1e2 * norm_1e4)#[(x * exp_normf)/(norm_1e2 * norm_1e4) for x in exp_inten]
        
        scanKey = str( exp_scan )+"."+str( exp_prech )+"."+str( exp_prec_mz )
        
        # mz_cols2, np_arr2
        delta_mz1 = ms1_tol*exp_prec_mz*1e-6
        left_lim1=exp_prec_mz-delta_mz1
        right_lim1=exp_prec_mz+delta_mz1
        matched_lib_data = libDF.loc[(libDF['prec_MZ'].astype("float") >= left_lim1) & (libDF['prec_MZ'].astype("float") <= right_lim1)]
        
        mz_cols2 = list(matched_lib_data.columns)
        np_arr2 = matched_lib_data.to_numpy()
        
        # frag_index
        frag_index = create_fragment_index(matched_lib_data)
        # all_frag
        all_frag = np.array([x for x in frag_index.keys()])
        # candidate_list
        candidate_list = get_lib_candidates(frag_index,all_frag,exp_prec_mz,exp_mz,ms1_tol,ms2_tol,match_threshold)
        if len(candidate_list)==0:
            unmatched_precursors+=1
            continue
        
        # 1.1. scan each matched_lib_data
        psm_score=np.zeros((len(candidate_list),5))
        pno=0
        max_score=0
        max_row2={}
        max_idx=0
        candino[cnt-1][0]=exp_scan
        candino[cnt-1][1]=len(candidate_list)
        
        for cno in candidate_list:
            row2 = np_arr2[cno]
            
            # load pkl (lib_data.columns: Index(['scan', 'charge', '[M+H]+', 'prec_MZ', 'L_ID', 'L_peptide', 'L_protein', 'RT', 'm/z', 'intensity', 'norm_factor']) )
            lib_normf = row2[mz_cols2.index("norm_factor")]
            lib_mz = np.array(row2[mz_cols2.index("m/z")])
            lib_inten = np.array(row2[mz_cols2.index("intensity")])
            lib_inten2=lib_inten*lib_normf/(norm_1e2*norm_1e4)
            # lib_inten2=np.array([(x * lib_normf)/(norm_1e2 * norm_1e4) for x in row2[mz_cols2.index("intensity")]])
            L_peptide = row2[mz_cols2.index("L_peptide")]
            if "Decoy_" in L_peptide.split(";")[0]:
                c_seq_only = get_pep_seq_only(L_peptide.split(";")[0].split("_")[1])
            else:
                c_seq_only = get_pep_seq_only(L_peptide.split(";")[0])
            
            # 1.1.1. scan each exp_inten
            c_match_num=0
            e_inten_sum=0
            r_inten_sum=0
            c_entropy_sim=0
            e_intens=[]
            r_intens=[]
            # e_fulllib_intens=[]
            # r_fulllib_intens=[]
            for ino in range(0,len(lib_inten)):
                if ms2_tol>1.0:
                    delta_mz2 = ms2_tol*lib_mz[ino]*1e-6
                else:
                    delta_mz2 = ms2_tol
                left_lim2 = lib_mz[ino]-delta_mz2
                right_lim2 = lib_mz[ino]+delta_mz2
                pos=np.nonzero((exp_mz>=left_lim2) & (exp_mz<=right_lim2))[0]
                if len(pos)==0:
                    # e_fulllib_intens.append(0)
                    # r_fulllib_intens.append(lib_inten[ino])
                    continue
                
                c_match_num+=1
                idx=np.argmax(exp_inten[pos])
                e_inten_sum+=exp_inten[pos[idx]]
                r_inten_sum+=lib_inten[ino]
                c_entropy_sim+=entropy_sim_per_peak(exp_inten2[pos[idx]],lib_inten2[ino])
                
                e_intens.append(exp_inten[pos[idx]])
                r_intens.append(lib_inten[ino])
                # e_fulllib_intens.append(exp_inten[pos[idx]])
                # r_fulllib_intens.append(lib_inten[ino])
            e_intens=np.array(e_intens, dtype=np.float64)
            r_intens=np.array(r_intens, dtype=np.float64)
            # e_fulllib_intens=np.array(e_fulllib_intens, dtype=np.float64)
            # r_fulllib_intens=np.array(r_fulllib_intens, dtype=np.float64)
            
            # 1.1.2. psm_score
            if c_match_num==0:
                c_hyperscore = 0
            else:
                # c_match_num = min([c_match_num,len(factorial)-1])
                c_match_num = min([c_match_num,len(log10factorial)-1])
                if method == "normalized_dot_product":
                    c_hyperscore = get_similarity(e_intens,r_intens)*len(e_intens)/len(lib_mz)
                elif method == "entropy":
                    c_hyperscore = c_entropy_sim
                elif method == "hyperscore":
                    # c_hyperscore = math.log(max([factorial[c_match_num]*factorial[c_match_num]*e_inten_sum*r_inten_sum,1]),10)
                    c_hyperscore = 2*log10factorial[c_match_num]+math.log(max([e_inten_sum,1]),10)+math.log(max([r_inten_sum,1]),10)
                else:
                    c_hyperscore = 0
            psm_score[pno][0]=c_hyperscore
            psm_score[pno][1]=len(e_intens)
            psm_score[pno][2]=len(lib_mz)
            psm_score[pno][3]=len(e_intens)/len(lib_mz)
            psm_score[pno][4]=len(c_seq_only)
            pno+=1
            
            # 1.1.3. max_score
            if max_score<c_hyperscore:
                max_score=c_hyperscore
                max_row2=row2
                max_idx=pno-1
        
        # ++++get_delta_cn++++
        [delta_cn,high2low_rank]=get_delta_cn_rank(psm_score)
        
        if method == "normalized_dot_product":
            psm_max = psm_score[max_idx]
            if psm_max[1]>=match_cutoff and psm_max[0]/psm_max[3]>=score_cutoff and psm_max[3]>=coverage_cutoff and min([psm_max[2],psm_max[4]])>=peplen_cutoff:
                L_peptide = max_row2[mz_cols2.index("L_peptide")]
                if "Decoy_" in L_peptide.split(";")[0]:# "decoy"
                    psm_score[max_idx][0] = psm_max[0]/psm_max[3]*(psm_max[3]+0)/2
                else:# "target"
                    psm_score[max_idx][0] = psm_max[0]/psm_max[3]*(psm_max[3]+1)/2
        
        # 1.2. top n matches
        if max_score==0:
            unmatched_precursors+=1
            continue
        
        for ino in range(0,len(psm_score)):
            cur_score = psm_score[ino][0]
            if cur_score<=0.05 or high2low_rank[ino]>=5:
                continue
            
            # ++++cur_delta_cn, cur_matched_num, cur_total_num++++
            cur_delta_cn = delta_cn[ino][0]
            cur_matched_num = psm_score[ino][1]
            cur_total_num = psm_score[ino][2]
            
            row2 = np_arr2[candidate_list[ino]]
            L_ID = row2[mz_cols2.index("L_ID")]
            L_RT = row2[mz_cols2.index("RT")]
            L_peptide = row2[mz_cols2.index("L_peptide")]
            # L_ID+";"+str(dp)+";"+str(L_RT)+";"+L_peptide
            # c_result = L_ID+";"+str(cur_score)+";"+str(L_RT)+";"+L_peptide
            c_result = f"{L_ID};{cur_score};{L_RT};{L_peptide};{cur_delta_cn};{cur_matched_num};{cur_total_num}"
            
            if scanKey not in dot_product_results.keys():
                dot_product_results[scanKey] = [c_result]
            else:
                dot_product_results[scanKey].append(c_result)
    
    
    
    
    '''

    if tolerance_type.upper() == "DYNAMIC":
        ms2_tol = max(int_sd_dict.values())

    for row in np_arr:
        scan = str(row[mz_cols.index("scan")])
        charge = int(row[mz_cols.index("charge")])
        precMZ = float(row[mz_cols.index("prec_MZ")])
        mz = row[mz_cols.index("m/z")]
        # bin_prec_mz = row[mz_cols.index("bin_prec_mz")]

        # reduced_library = libDF.loc[libDF["bin_prec_mz"] == bin_prec_mz]

        if method == "normalized_dot_product":

            intensity = list(row[mz_cols.index("intensity")])
        else:
            intensity = list(row[mz_cols.index("normalized_intensity")])
        
        if tolerance_type.upper() == "DYNAMIC":
            log10_intensity = list(row[mz_cols.index("log10_intensity")]) #this intensity is used for dynamic tolerance
        

        matched_lib_DF = scanPrecursorMatch(precMZ, charge, libDF, ms1_tol)
        

        cnt+=1

        if matched_lib_DF.shape[0] >=1: #there may not be any matches to the library so we have to check the library match to avoid error
    #         print ("scan number is ",scan)
    #         dfDict[scan] = matched_lib_DF
            if top_ions_control == "1": #now this means program selects top n_ions from library and makes sure that these are present in the spectrum 
                # matched_lib_DF_top2 = select_TopN_Lib_ions(mz, intensity, matched_lib_DF, n_ions, ms2_tol, topRanks) 
                matched_lib_DF_top2 = checkTopLibraryIons(mz, matched_lib_DF, min_top_ions, top_ions,ms2_tol)
                # matched_lib_DF_top2 = checkTopLibraryIons(mz, matched_lib_DF, min_top_ions, top_ions,ms2_tol)

            else:
                matched_lib_DF_top2 = matched_lib_DF

            if str(binsize).upper() == "NO":
                spectrumInputDict = {"mz":mz,"intensity":intensity}
                if tolerance_type.upper() == "DYNAMIC":
                    tolInputDict = {"mz":mz,"intensity":log10_intensity}
                else:
                    tolInputDict = {}
            else:
                binsize = int(binsize)
                spectrumInputDict = binning_mz_100(mz, intensity, top_ions_per_bin,binsize) #simplified dictionary with 10 top ions for each 100 mz bins with normalized intensity 
                #this is the input for dot product that will be compared with library

                #the input dictionary for tolerance dynamic tolerance
                if tolerance_type.upper() == "DYNAMIC":
                    tolInputDict = binning_mz_100(mz, log10_intensity, top_ions_per_bin,binsize) #SD input file has intensity as log10 so we need log10_intensity
                else:
                    tolInputDict = {}


            

            mz_cols2 = list(matched_lib_DF_top2.columns)
            np_arr2 = matched_lib_DF_top2.to_numpy()
            for row2 in np_arr2: 
                #removal of library cleaning step enables us to directly generate lib_mz_int_dict
                #lib_mz_int_dict2 = {}
                lib_mz_int_dict = {}
                L_ID = row2[mz_cols2.index("L_ID")]
                RT = row2[mz_cols2.index("RT")]
                L_peptide = row2[mz_cols2.index("L_peptide")]
                prec_mz = row2[mz_cols2.index("prec_MZ")]
    #             print(L_peptide)

                #library cleaning step is removed so lib_mz_int_dict2 can be saved as lib_mz_int_dict

                lib_mz_int_dict["mz"] = row2[mz_cols2.index("m/z")]

                # print ("The library entry is {} and peptide is {}".format(L_ID, L_peptide))
                # print ("Length of library product ions = {}".format(len(lib_mz_int_dict2["mz"])))

                if method == "normalized_dot_product":
                    lib_mz_int_dict["intensity"] = list(row2[mz_cols2.index("intensity")]) #use normalize dictionary for the library input
                else:
                    lib_mz_int_dict["intensity"] = list(row2[mz_cols2.index("normalized_intensity")]) #use normalize dictionary for the library input            
    #             #tolerance_type defines how to choose the tolerance Dynamic will use the int_sd_dict dictionary else static tolerace will be used
                #clean library ions too to check for ions within mass tolerance           
                

                #may be this is not required as we have all theoretical ions
                #lib_mz_int_dict = cleanLibRedundancy(lib_mz_int_dict2, ms2_tol)
                
                # print (".... The test scan is {}".format(scan))

                # if scan == "1765":
                #     print ("spectrum input dict = {}".format(spectrumInputDict))
                #     print ("tolInputDict = {}".format(tolInputDict))
                #     print ("int_sd_dict = {}".format(int_sd_dict))
                #     print ("tolerance_type = {}".format(tolerance_type))
                #     print ("lib_mz_int_dict = {}".format(lib_mz_int_dict))


                tr_featSpec = trimFeatSpec(spectrumInputDict,tolInputDict,int_sd_dict,tolerance_type,lib_mz_int_dict,ms2_tol) #int_sd_dict = tolerance SD defined intensity from the file after mass calibration
                #dp = calcMS2Similarity(tr_featSpec,lib_mz_int_dict,ms2_tol)
                #input for other scoring techniques
                spec_query = np.array(tr_featSpec["intensity"])
                spec_reference = np.array(lib_mz_int_dict["intensity"])
                #print (spec_query)
                #print (spec_reference)

                #if method == "JiHoon_Norm_DP":
                #    dp = calcMS2Similarity(tr_featSpec,lib_mz_int_dict,ms2_tol)


                if method == "normalized_dot_product":
                    dp = normalizedDotProduct(spec_query,spec_reference)

                else:
                    spec_query_ = conversionDictSpecToNumpyArrayFormat(tr_featSpec)
                    spec_reference_ = conversionDictSpecToNumpyArrayFormat(lib_mz_int_dict)
            
                    if method == "DoubleNormalizedDP":
                        dp = normalizedDotProduct(spec_query,spec_reference)
                    if method == "DP_Peng":
                        dp = DP_Peng_similarity(spec_query,spec_reference)
                    if method == "unweighted_entropy":
                        dp = unweightedEntropySimCalc(spec_query_,spec_reference_)
                    if method == "dot_product":
                        dp = dot_product_similarity(spec_query,spec_reference)
                    if method == "fidelity":
                        dp = fidelity_similarity(spec_query,spec_reference)
                    if method == "bhattacharya_2":
                        dp = bhattacharya_2_similarity(spec_query,spec_reference)

                # if "Decoy_" in L_peptide:
                #     L_peptide2 = L_peptide.split(";")
                #     L_peptide = "{};{};{}".format(L_peptide2[0],L_peptide2[1],prec_mz)
                scanKey = str(scan)+"."+str(charge)+"."+str(precMZ)

                if dp > 0.05:
                    if scanKey not in dot_product_results.keys():
                        dot_product_results[scanKey] = [L_ID+";"+str(dp)+";"+str(RT)+";"+L_peptide]
                    else:
                        dot_product_results[scanKey].append(L_ID+";"+str(dp)+";"+str(RT)+";"+L_peptide)
        else:
            unmatched_precursors+=1       
        
    '''
    if cnt == update:
        remainingScans = len(expDF) - cnt
        print ("Total scan searched = ", cnt,"\nRemaining scans = " ,remainingScans)
        update+=10000
        
    #     else:
    #         dot_product_results[int(scan)] = ["NA;0.0;0.0;Decoy_fake_NA"]
    final_result = {}
    for results in dot_product_results.keys():
    #     print (len(dot_product_results[results]))
        final_result[results] = ",".join(dot_product_results[results])
    print (".......Total unmatched precursors to the library within the {} ppm tolerance = {} ".format(ms1_tol, unmatched_precursors)) 
    return final_result


#this function is used to perform QC analysis for after correction matches 

def afterCorrQC(expMZXML,expDF,libDF,top_ions,ms2_tol,ms1_tol,top_ions_per_bin,binsize,tolerance_type,int_sd_dict):
    # dfDict = {}#collection of matched precursor MZ dataframe
    # ms2_tol = 15 #this is parameter for precursor ion matches
    QC_dictionary = {} #initialize empty dictionary for QC
    matchedIonList = [] #For QC
    spectrumList = [] #For QC
    top_ion_match_list = [] #For QC

    mz_cols = list(expDF.columns)
    np_arr = expDF.to_numpy()
    # top_ion=3
    dot_product_results = {}
    cnt = 0
    update = 5000
    if tolerance_type.upper() == "DYNAMIC":
        ms2_tol = max(int_sd_dict.values())

    for row in np_arr:
        scan = str(row[mz_cols.index("scan")])
        charge = int(row[mz_cols.index("charge")])
        precMZ = float(row[mz_cols.index("prec_MZ")])
        mz = row[mz_cols.index("m/z")]
        intensity = list(row[mz_cols.index("normalized_intensity")])
        log10_intensity = list(row[mz_cols.index("log10_intensity")]) #this intensity is used for dynamic tolerance
        #print ("precursor m/z = ",precMZ)
        spectrum = expMZXML+"."+scan+"."+str(charge)


        matched_lib_DF = scanPrecursorMatch(precMZ, charge, libDF, ms1_tol)
        #print (matched_lib_DF) 
        cnt+=1

        if matched_lib_DF.shape[0] >=1: #there may not be any matches to the library so we have to check the library match to avoid error
    #         print ("scan number is ",scan)
    #         dfDict[scan] = matched_lib_DF
            matched_lib_DF_top2 = checkTopLibraryIons(mz, matched_lib_DF, min_top_ions, top_ions,ms2_tol)
            if str(binsize).upper() == "NO":
                spectrumInputDict = {"mz":mz,"intensity":intensity}
                tolInputDict = {"mz":mz,"intensity":log10_intensity}
            else:
                binsize = int(binsize)
                spectrumInputDict = binning_mz_100(mz, intensity, top_ions_per_bin,binsize) #simplified dictionary with 10 top ions for each 100 mz bins with normalized intensity 
                #this is the input for dot product that will be compared with library

                #the input dictionary for tolerance dynamic tolerance
                tolInputDict = binning_mz_100(mz, log10_intensity, top_ions_per_bin,binsize) #SD input file has intensity as log10 so we need log10_intensity

            mz_cols2 = list(matched_lib_DF_top2.columns)
            np_arr2 = matched_lib_DF_top2.to_numpy()
            for row2 in np_arr2: 
                lib_mz_int_dict = {}
                
                top_ions_match = int(row2[mz_cols2.index("topIonsExist")]) #For QC
                scanLib = str(row[mz_cols2.index("scan")]) #For QC
                chargeLib = int(row[mz_cols2.index("charge")]) #For QC
                spectrumLib = expMZXML+"."+scanLib+"."+str(chargeLib)

                lib_mz_int_dict["mz"] = row2[mz_cols2.index("m/z")]
                lib_mz_int_dict["intensity"] = list(row2[mz_cols2.index("normalized_intensity")]) #use normalize dictionary for the library input

    #             #tolerance_type defines how to choose the tolerance Dynamic will use the int_sd_dict dictionary else static tolerace will be used
                
                if spectrum == spectrumLib:
                    tr_featSpec, matchedIonsNo = trimFeatSpecQC(spectrumInputDict,tolInputDict,int_sd_dict,tolerance_type,lib_mz_int_dict) #int_sd_dict = tolerance SD defined intensity from the file after mass calibration
                
                    matchedIonList.append(matchedIonsNo) #For QC
                    spectrumList.append(spectrum) #For QC
                    top_ion_match_list.append(top_ions_match) #For QC

                
    if cnt == update:
        remainingScans = len(expDF) - cnt
        print ("Total scan searched = ", cnt,"\nRemaining scans = " ,remainingScans)
        update+=5000
            
    
    QC_dictionary["spectrum"] = spectrumList #For QC
    QC_dictionary["matchedIonPostCorr"] = matchedIonList #For QC
    QC_dictionary["TopIonCountPostCorr"] = top_ion_match_list #For QC

    qcDF = pd.DataFrame(QC_dictionary) #For QC
    qcDF.to_csv("QC_afterMassCorrection_"+tolerance_type+".csv") #For QC



##################TWO scores specific################

def search(expDF,libDF,n_ions, ms2_tol,ms1_tol, int_sd_dict,topRanks,method="normalized_dot_product"):
    
    mz_cols = list(expDF.columns)
    np_arr = expDF.to_numpy()

    dot_product_results = {}
    cnt = 0
    update = 10000
    
    for row in np_arr:
        scan = str(row[mz_cols.index("scan")])
        charge = int(row[mz_cols.index("charge")])
        precMZ = float(row[mz_cols.index("prec_MZ")])
        mz = row[mz_cols.index("m/z")]
        
        intensity = list(row[mz_cols.index("intensity")])
        matched_lib_DF = scanPrecursorMatch(precMZ, charge, libDF, ms1_tol)
        
        
        if matched_lib_DF.shape[0] >=1: #there may not be any matches to the library so we have to check the library match to avoid error
            matched_lib_DF_top2 = select_TopN_Lib_ions(mz, intensity, matched_lib_DF, n_ions, ms2_tol, topRanks)
            spectrumInputDict = {"mz":mz,"intensity":intensity}
            tolInputDict = {}
            tolerance_type = "Static"
            cnt+=1
            
            mz_cols2 = list(matched_lib_DF_top2.columns)
            np_arr2 = matched_lib_DF_top2.to_numpy()
            for row2 in np_arr2: 
                #removal of library cleaning step enables us to directly generate lib_mz_int_dict
                #lib_mz_int_dict2 = {}
                lib_mz_int_dict = {}
                L_ID = row2[mz_cols2.index("L_ID")]
                RT = row2[mz_cols2.index("RT")]
                L_peptide = row2[mz_cols2.index("L_peptide")]
                prec_mz = row2[mz_cols2.index("prec_MZ")]
    #             print(L_peptide)

                #library cleaning step is removed so lib_mz_int_dict2 can be saved as lib_mz_int_dict

                lib_mz_int_dict["mz"] = row2[mz_cols2.index("m/z")]

                # print ("The library entry is {} and peptide is {}".format(L_ID, L_peptide))
                # print ("Length of library product ions = {}".format(len(lib_mz_int_dict2["mz"])))

                if method == "normalized_dot_product":
                    lib_mz_int_dict["intensity"] = list(row2[mz_cols2.index("intensity")]) #use normalize dictionary for the library input
                else:
                    lib_mz_int_dict["intensity"] = list(row2[mz_cols2.index("normalized_intensity")]) #use normalize dictionary for the library input            
    #             #tolerance_type defines how to choose the tolerance Dynamic will use the int_sd_dict dictionary else static tolerace will be used
                #clean library ions too to check for ions within mass tolerance           
                

                #may be this is not required as we have all theoretical ions
                #lib_mz_int_dict = cleanLibRedundancy(lib_mz_int_dict2, ms2_tol)
                

                tr_featSpec = trimFeatSpec(spectrumInputDict,tolInputDict,int_sd_dict,tolerance_type,lib_mz_int_dict,ms2_tol) #int_sd_dict = tolerance SD defined intensity from the file after mass calibration
                #dp = calcMS2Similarity(tr_featSpec,lib_mz_int_dict,ms2_tol)
                #input for other scoring techniques
                spec_query = np.array(tr_featSpec["intensity"])
                spec_reference = np.array(lib_mz_int_dict["intensity"])
                #print (spec_query)
                #print (spec_reference)

                #if method == "JiHoon_Norm_DP":
                #    dp = calcMS2Similarity(tr_featSpec,lib_mz_int_dict,ms2_tol)


                if method == "normalized_dot_product":
                    dp = normalizedDotProduct(spec_query,spec_reference)

                else:
                    spec_query_ = conversionDictSpecToNumpyArrayFormat(tr_featSpec)
                    spec_reference_ = conversionDictSpecToNumpyArrayFormat(lib_mz_int_dict)
            
                    if method == "DoubleNormalizedDP":
                        dp = normalizedDotProduct(spec_query,spec_reference)
                    if method == "DP_Peng":
                        dp = DP_Peng_similarity(spec_query,spec_reference)
                    if method == "unweighted_entropy":
                        dp = unweightedEntropySimCalc(spec_query_,spec_reference_)
                    if method == "dot_product":
                        dp = dot_product_similarity(spec_query,spec_reference)
                    if method == "fidelity":
                        dp = fidelity_similarity(spec_query,spec_reference)
                    if method == "bhattacharya_2":
                        dp = bhattacharya_2_similarity(spec_query,spec_reference)

                if "Decoy_" in L_peptide:
                    L_peptide2 = L_peptide.split(";")
                    L_peptide = "{};{};{}".format(L_peptide2[0],L_peptide2[1],prec_mz)
                scanKey = str(scan)+"."+str(charge)+"."+str(precMZ)


                if scanKey not in dot_product_results.keys():
                    dot_product_results[scanKey] = [L_ID+";"+str(dp)+";"+str(RT)+";"+L_peptide]
                else:
                    dot_product_results[scanKey].append(L_ID+";"+str(dp)+";"+str(RT)+";"+L_peptide)
                     
        
        if cnt == update:
            remainingScans = len(expDF) - cnt
            print ("Total scan searched = ", cnt,"\nRemaining scans = " ,remainingScans)
            update+=10000
       
    final_result = {}
    for results in dot_product_results.keys():
    #     print (len(dot_product_results[results]))
        final_result[results] = ",".join(dot_product_results[results])
#     print (final_result) 
    return final_result


#heckMaxLibMZ = np.max(libSpec["mz"])
    # checkMinLibMZ = np.min(libSpec["mz"])

    # #to avoid unnecessary tolerance calculation directly use the maximum and min ions in library 
    # #with some tolerance to cutoff unwanted spectra
    # max_exp_mzCheck = float(checkMaxLibMZ)+(checkMaxLibMZ/1000000*tol_max)
    # min_exp_mzCheck = float(checkMinLibMZ)-(checkMinLibMZ/1000000*tol_max)

def select_TopN_Lib_ions(exp_mz_list, intensity_exp, matched_library_DF, n_ions, ms2_tol, topRanks=10): #top_ion is the minimum no of ions (intensity ranked top) that should be present in experimental mz 
    
    
    dotProductList = [] #this checks if top_ion are present in exp_mz_list
    
    #this is to speed up the scanning process as numpy array are fastest
    mz_cols = list(matched_library_DF.columns) #for indexing columns
    np_arr = matched_library_DF.to_numpy() #numpy array conversion of dataframe
    for row in np_arr: #each matched library checked for top ions according to parameters above
        mz = row[mz_cols.index("m/z")] #mz list (ms2)
        intensity = row[mz_cols.index("intensity")] #intensity list (ms2)
        scan = str(row[mz_cols.index("scan")]) #scan 
       
        ind = np.argsort([-i for i in intensity]) #sorting by descending order intensity
        

        top_ion_lib = {"mz":[],"intensity":[]} #empty dict for top ion collection
        
        top_ion_lib["mz"] = [mz[i] for i in ind[0:n_ions]]
        top_ion_lib["intensity"] = [intensity[i] for i in ind[0:n_ions]]
        
        
        cnt = 0
        #make a trimeed spectral dictionary and update it as the ion is within the tolerance
        tr_featSpec = {}

        checkMaxLibMZ = np.max(top_ion_lib["mz"])
        checkMinLibMZ = np.min(top_ion_lib["mz"])

        max_exp_mzCheck = float(checkMaxLibMZ)+(checkMaxLibMZ/1000000*ms2_tol)
        min_exp_mzCheck = float(checkMinLibMZ)-(checkMinLibMZ/1000000*ms2_tol)

        # if (val <= max_exp_mzCheck) and (val >= min_exp_mzCheck):
        for ion in top_ion_lib["mz"]: #looping over top_ions to see if they are present in experimental mz
            for index, masses in enumerate(exp_mz_list):
                if (masses <= max_exp_mzCheck) and (masses >= min_exp_mzCheck):
                    if ion not in tr_featSpec.keys():
                        tr_featSpec[ion]=intensity_exp[index]
                    
                        
                    else:
                        old_intensity = tr_featSpec[ion]
                        
                        if intensity_exp[index] > old_intensity:
                            tr_featSpec[ion] = intensity_exp[index] 
        
        
#         print ("Top library ions ",top_ion_lib["mz"])
        
        tr_featSpec2 = {"mz":[],"intensity":[]}
        
        for ions in top_ion_lib["mz"]:
            #add intensity 0 to all the ions that are present in library but are not within the given tolerance in spectrum raw
            if ions in tr_featSpec.keys():
                tr_featSpec2["mz"].append(ions)
                tr_featSpec2["intensity"].append(tr_featSpec[ions])
            else:
                tr_featSpec2["mz"].append(ions)
                tr_featSpec2["intensity"].append(0.0)

#         print (tr_featSpec2)
        
        #compute normalized dot product here
        spec_query = np.array(tr_featSpec2["intensity"])
        spec_reference = np.array(top_ion_lib["intensity"])
        
        dp = normalizedDotProduct(spec_query,spec_reference)
        dotProductList.append(dp)
        
    matched_library_DF["QuickDotProduct"] = dotProductList
    #Select top 10 Ranks
    matched_lib_DF_top = matched_library_DF.sort_values(by=["QuickDotProduct"], ascending=False)
        
    return matched_lib_DF_top.iloc[0:topRanks]

