import pandas as pd
import pyteomics
from pyteomics import mass
from pyteomics import mzxml,mzml
import numpy as np
import os, sys
from collections import Counter
import pickle
import statsmodels.api as sm
import re
from elutionCases import *
from process_ms3 import *


class progressBar:
    def __init__(self, total):
        self.total = total
        self.barLength = 20
        self.count = 0
        self.progress = 0
        self.block = 0
        self.status = ""

    def increment(self, nIncrement=None):
        if nIncrement == None:
            self.count += 1
        else:
            self.count = nIncrement
        self.progress = self.count / self.total
        self.block = int(round(self.barLength * self.progress))
        if self.progress == 1:
            self.status = "Done...\r\n"
        else:
            self.status = ""
        #         self.status = str(self.count) + "/" + str(self.total)
        text = "\r  Progress: [{0}] {1}% {2}".format("#" * self.block + "-" * (self.barLength - self.block),
                                                     int(self.progress * 100), self.status)
        sys.stdout.write(text)
        sys.stdout.flush()



def mkdir(outputFolder):
    #create search output directory
    cmdDir = "mkdir "+outputFolder
    try:
        os.system(cmdDir)
    except:
        write_log ("Directory exist")


def mzxml_2_df(mzFILE,raw_data_type):
    #mzXcols = ["num","msLevel","peaksCount","retentionTime","msType","activationMethod","precursorMz","precursorCh","m/z array","intensity array","precursorIntensity","basePeakIntensity"]
    if raw_data_type=="mzXML":
        data = []
        reader = mzxml.MzXML(mzFILE)
        
        ms3level = 0
        for ino in range(0,100):
            spec = reader[ino]
            msLevel = int(spec["msLevel"])
            if msLevel==3:
                ms3level = 1
            if ms3level==1:
                break
        
        ms123scan_list = []
        ms123scan_ms2_array = np.array([])
        ms123scan_ms3_list = []
        if ms3level==1:
            [ms123scan_list, ms123scan_ms2_array, ms123scan_ms3_list] = Get_ms123scan_list_via_mzXML(reader)
        
        for ino in range(0,len(reader)):
            spec = reader[ino]
            num = int(spec["num"])
            msLevel = int(spec["msLevel"])
            peaksCount = int(spec["peaksCount"])
            retentionTime = float(spec["retentionTime"]) # min
            
            if msLevel==3 or (msLevel==2 and ms3level==1 and num not in ms123scan_ms2_array):
                continue
            
            try:
                msType = spec["filterLine"][0:4]
            except:
                msType = "FTMS"
            try:
                activationMethod = spec["precursorMz"][0]["activationMethod"]
            except:
                activationMethod = "HCD"
            if msLevel==1:
                precursorMz = 0.0
            else:
                try:
                    filterLine = spec["filterLine"] # converted from ReAdW
                except:
                    filterLine = "-" # converted from msconvert
                if filterLine != "-":
                    precursorMz = float(re.search(r"ms2 ([0-9.]+)\@", spec["filterLine"]).group(1)) # converted from ReAdW
                else:
                    precursorMz = int(float(spec["precursorMz"][-1]["precursorMz"])*1e4+0.4)/1e4 # converted from msconvert
            try:
                precursorCh = int(spec["precursorMz"][0]["precursorCharge"])
            except:
                precursorCh = 2
            mz_array = spec["m/z array"]
            intensity_array = spec["intensity array"]
            try:
                precursorIntensity = float(spec["precursorMz"][0]["precursorIntensity"])
            except:
                precursorIntensity = 0.0
            basePeakIntensity = float(spec["basePeakIntensity"])
            
            if msLevel==2 and ms3level==1 and num in ms123scan_ms2_array:
                pos1=np.nonzero( ms123scan_ms2_array==num )[0]
                if len(pos1)>0:
                    ms3scan = ms123scan_ms3_list[pos1[0]]
                    ms3_spec = reader[ms3scan-1]
                    ms3mz = ms3_spec["m/z array"]
                    ms3inten = ms3_spec["intensity array"]
                    [mz_array,intensity_array] = Merge_ms23(mz_array,intensity_array,ms3mz,ms3inten)
                    peaksCount = len(mz_array)
            
            spectrum_data = {
                "num": num,
                "msLevel": msLevel,
                "peaksCount": peaksCount,
                "retentionTime": retentionTime,
                "msType": msType,
                "activationMethod": activationMethod,
                "precursorMz": precursorMz,
                "precursorCh": precursorCh,
                "m/z array": mz_array,
                "intensity array": intensity_array,
                "precursorIntensity": precursorIntensity,
                "basePeakIntensity": basePeakIntensity
            }
            
            data.append(spectrum_data)
        
        dfmzXML = pd.DataFrame(data)
        
        # ms1 = dfmzXML.loc[dfmzXML.msLevel==1]     #ms1 level scans
        # np_arr1 = ms1.to_numpy()
        # ms2 = dfmzXML.loc[dfmzXML.msLevel==2]     #ms2 level scans
        # np_arr2 = ms2.to_numpy()
    elif raw_data_type=="mzML":
        data = []
        reader = mzml.MzML(mzFILE)
        
        ms3level = 0
        for ino in range(0,100):
            spec = reader[ino]
            msLevel = spec['ms level']
            if msLevel==3:
                ms3level = 1
            if ms3level==1:
                break
        
        ms123scan_list = []
        ms123scan_ms2_array = np.array([])
        ms123scan_ms3_list = []
        if ms3level==1:
            [ms123scan_list, ms123scan_ms2_array, ms123scan_ms3_list] = Get_ms123scan_list_via_mzML(reader)
        
        for ino in range(0,len(reader)):
            spec = reader[ino]
            num = int(spec['id'].split('scan=')[-1])
            msLevel = spec['ms level']
            peaksCount = int(spec['defaultArrayLength'])
            retentionTime = spec['scanList']['scan'][0]['scan start time'] # min
            
            if msLevel==3 or (msLevel==2 and ms3level==1 and num not in ms123scan_ms2_array):
                continue
            
            try:
                msType = spec['scanList']['scan'][0]['filter string'][0:4]
            except:
                msType = "FTMS"
            try:
                activationMethod = re.search(r"@(\D+)", spec['scanList']['scan'][0]['filter string']).group(1).upper()
            except:
                activationMethod = "HCD"
            if msLevel==1:
                precursorMz = 0.0
            else:
                try:
                    precursorMz = float(re.search(r"ms2 ([0-9.]+)\@", spec['scanList']['scan'][0]['filter string']).group(1))
                except:
                    precursorMz = int(spec['precursorList']['precursor'][-1]['selectedIonList']['selectedIon'][0]['selected ion m/z']*1e4+0.4)/1e4
            try:
                precursorCh = spec['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['charge state']
            except:
                precursorCh = 2
            mz_array = spec["m/z array"]
            intensity_array = spec["intensity array"]
            try:
                precursorIntensity = spec['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['peak intensity']
            except:
                precursorIntensity = 0.0
            basePeakIntensity = spec['base peak intensity']
            
            if msLevel==2 and ms3level==1 and num in ms123scan_ms2_array:
                pos1=np.nonzero( ms123scan_ms2_array==num )[0]
                if len(pos1)>0:
                    ms3scan = ms123scan_ms3_list[pos1[0]]
                    ms3_spec = reader[ms3scan-1]
                    ms3mz = ms3_spec["m/z array"]
                    ms3inten = ms3_spec["intensity array"]
                    [mz_array,intensity_array] = Merge_ms23(mz_array,intensity_array,ms3mz,ms3inten)
                    peaksCount = len(mz_array)
            
            spectrum_data = {
                "num": num,
                "msLevel": msLevel,
                "peaksCount": peaksCount,
                "retentionTime": retentionTime,
                "msType": msType,
                "activationMethod": activationMethod,
                "precursorMz": precursorMz,
                "precursorCh": precursorCh,
                "m/z array": mz_array,
                "intensity array": intensity_array,
                "precursorIntensity": precursorIntensity,
                "basePeakIntensity": basePeakIntensity
            }
            
            data.append(spectrum_data)
        
        dfmzXML = pd.DataFrame(data)
        
        # ms1 = dfmzXML.loc[dfmzXML.msLevel==1]     #ms1 level scans
        # np_arr1 = ms1.to_numpy()
        # ms2 = dfmzXML.loc[dfmzXML.msLevel==2]     #ms2 level scans
        # np_arr2 = ms2.to_numpy()
    else:
        data = {"num": [], "msLevel": [], "peaksCount": [], "retentionTime": [], "msType": [], "activationMethod": [], "precursorMz": [], "precursorCh": [], "m/z array": [], "intensity array": [], "precursorIntensity": [], "basePeakIntensity": []}
        dfmzXML = pd.DataFrame(data)
        # np_arr1 = []
        # np_arr2 = []
    
    return dfmzXML#,np_arr1,np_arr2

# this is just for ms2RT during library search
def get_ms2_rt_dict(mzxml,raw_data_type):
    df = mzxml_2_df(mzxml,raw_data_type)
    scans = list(df["num"])
    rt = list(df["retentionTime"])
    
    rt_dict = {}
    for index,scan in enumerate(scans):
        rt_dict[int(scan)] = float(rt[index])
    return rt_dict


def getMs2ToSurvey(mzxml,raw_data_type):
    df = mzxml_2_df(mzxml,raw_data_type)
    scans = list(df["num"])
    mslevel = list(df["msLevel"])
    rt = list(df["retentionTime"])
    
    all_ms1_scans = []
    
    print("  Read a mzxml file dataframe: to find survey scans of MS2 scans")
    res = {}
    rt_dict = {}
    for index,scan in enumerate(scans):
        rt_dict[int(scan)] = float(rt[index])
        if mslevel[index] == 1:
            survey = int(scan)
            all_ms1_scans.append(survey)
        elif mslevel[index] == 2:
            res[int(scan)] = survey
#     print("  Done ...\n")
    return res,rt_dict,all_ms1_scans, df




def parse_idtxt(idtxt):
    psms = pd.read_csv(idtxt, skiprows=1, sep=";")  # Note that ID.txt file is delimited by semicolon
    psms = psms[["Peptide", "Outfile", "XCorr","measuredMH", "calcMH"]]
    #sort by XCorr 

    psms["XCorr"] = psms["XCorr"].astype("float")
    psms.sort_values(by=["XCorr"], ascending=False)


#     psms = psms.loc[psms["Outfile"].str.contains(mzXMLBaseName)]    # Extract PSMs from FTLD_Batch2_F50.mzXML
    psms["charge"] = [outfile.split("/")[-1].split(".")[-2] for outfile in psms["Outfile"]]
    psms = psms.drop_duplicates()
    

    # Unique key is peptide-charge pair
    print("  RT of every identified peptide is being inferred and assigned")
    keys = psms["Peptide"] + "_" + psms["charge"]
    psms["keys"] = keys

    return psms



def get_df_rt_tol(dfMz, rt_lower, rt_higher):
    df = dfMz.loc[dfMz["msLevel"] == 1]
    df = df[(df["retentionTime"] >= rt_lower) & (df["retentionTime"] <= rt_higher)]
    ms1_scans_tol = list(df["num"])
    return ms1_scans_tol



def getPrecursorPeak(dfMz, surveyScanNumber,nominalPrecMz,  rt_higher, rt_lower, ms1_tol):
    # Find the precursor peak of the PSM (the strongest peak within the isolation window)
    #get ms1 informatio
    
    df = dfMz.loc[dfMz.num == str(surveyScanNumber)]
    
    df = df[(df["retentionTime"].astype("float") >= rt_lower) & (df["retentionTime"].astype("float") <= rt_higher)]
    
    
    mzArray = np.array(df["m/z array"].to_list()[0])
    intArray = np.array(df["intensity array"].to_list()[0])
    
    max_prec_mzCheck = nominalPrecMz+(nominalPrecMz/1000000*ms1_tol)
    min_prec_mzCheck = nominalPrecMz-(nominalPrecMz/1000000*ms1_tol)
    
    ind = (mzArray >= min_prec_mzCheck) & (mzArray <= max_prec_mzCheck)
    if sum(ind > 0):
        subMzArray = mzArray[ind]
        subIntArray = intArray[ind]
        ind2 = np.argmax(subIntArray)
        precMz = subMzArray[ind2]
        precIntensity = subIntArray[ind2]
    else:
        precMz = -1
        precIntensity = -1
    precRt = df["retentionTime"].values[0]   # Unit of minute
    
    return precMz, precIntensity, precRt



def get_rt_jdscore(psms, mzxml, raw_data_type): #psms = df of idtxt
    
    ms2ToSurvey,rt_dict,all_ms1_scans, dfMz = getMs2ToSurvey(mzxml, raw_data_type)
    
    mzXMLBaseName = os.path.basename(mzxml).split(".")[0]
    psms_run = psms.loc[psms["Outfile"].str.contains(mzXMLBaseName+"\.")]    # Extract PSMs from FTLD_Batch2_F50.mzXML
    
    keys = list(set(psms_run["keys"]))
    

    # list of outfiles
    proton = mass.calculate_mass(formula='H+') #proron mono mass

    psm_scan_list = []
    prec_mz_list = []
    prec_int_list = []
    rt_list = []
    keys_list = []
    jdscore_list = []

    mz_cols = list(psms_run.columns)
    
    
    prec_key_cnt = 0
    for key in keys:
        prec_key_cnt +=1 
        
        if prec_key_cnt%10000 == 0:
            print ("Total precursors {} analyzed out of {}".format(prec_key_cnt,len(keys)))
        
        psms_subset = psms_run.loc[psms_run["keys"]==key]
        
        
        np_arr = psms_subset.to_numpy()
        for row in np_arr:
            
            keys_list.append(key)
            outfiles = str(row[mz_cols.index("Outfile")])
            jdscore = str(row[mz_cols.index("JDscore")])
            [_, psmScanNum, _] = os.path.basename(outfiles).split(".")
            
            pep = str(row[mz_cols.index("Peptide")])
            z = str(row[mz_cols.index("z")])
            
            
            precRt = rt_dict[int(psmScanNum)]
            
            df = dfMz[dfMz["num"]==int(psmScanNum)]
            
            
            intArray = np.array(df["intensity array"].to_list()[0])
            prec_int = np.max(intArray)
            
            measuredMH = row[mz_cols.index("measuredMH")]
            nominalPrecMz = ((measuredMH - proton)+(int(z)*proton))/int(z)

                    
            psm_scan_list.append(psmScanNum)
            rt_list.append(precRt)
            prec_mz_list.append(nominalPrecMz)
            prec_int_list.append(prec_int)
            jdscore_list.append(jdscore)

#             print ("total MS1 checked = {}".format(cnt))
    out_table = pd.DataFrame({"peptide_charge":keys_list,"ms2_scan":psm_scan_list,"prec_mz":prec_mz_list,"prec_intensity":prec_int_list,"ms2_rt":rt_list, "JDscore":jdscore_list})
    
    return out_table


def get_rt(psms, mzxml, raw_data_type): #psms = df of idtxt
    
    ms2ToSurvey,rt_dict,all_ms1_scans, dfMz = getMs2ToSurvey(mzxml, raw_data_type)
    
    mzXMLBaseName = os.path.basename(mzxml).split(".")[0]
    psms_run = psms.loc[psms["Outfile"].str.contains(mzXMLBaseName+"\.")]    # Extract PSMs from FTLD_Batch2_F50.mzXML
    
    keys = list(set(psms_run["keys"]))
    

    # list of outfiles
    proton = mass.calculate_mass(formula='H+') #proron mono mass

    psm_scan_list = []
    prec_mz_list = []
    prec_int_list = []
    rt_list = []
    keys_list = []
    
    mz_cols = list(psms_run.columns)
    
    
    prec_key_cnt = 0
    for key in keys:
        prec_key_cnt +=1 
        
        if prec_key_cnt%10000 == 0:
            print ("Total precursors {} analyzed out of {}".format(prec_key_cnt,len(keys)))
        
        psms_subset = psms_run.loc[psms_run["keys"]==key]
        
        
        np_arr = psms_subset.to_numpy()
        for row in np_arr:
            
            keys_list.append(key)
            outfiles = str(row[mz_cols.index("Outfile")])
            [_, psmScanNum,_ , _] = os.path.basename(outfiles).split(".")
            
            pep = str(row[mz_cols.index("Peptide")])
            z = str(row[mz_cols.index("z")])
            
            
            precRt = rt_dict[int(psmScanNum)]
            
            df = dfMz[dfMz["num"]==int(psmScanNum)]
            
            
            intArray = np.array(df["intensity array"].to_list()[0])
            prec_int = np.max(intArray)
            
            measuredMH = row[mz_cols.index("measuredMH")]
            nominalPrecMz = ((measuredMH - proton)+(int(z)*proton))/int(z)

                    
            psm_scan_list.append(psmScanNum)
            rt_list.append(precRt)
            prec_mz_list.append(nominalPrecMz)
            prec_int_list.append(prec_int)
            

#             print ("total MS1 checked = {}".format(cnt))
    out_table = pd.DataFrame({"peptide_charge":keys_list,"ms2_scan":psm_scan_list,"prec_mz":prec_mz_list,"prec_intensity":prec_int_list,"ms2_rt":rt_list})
    
    return out_table


#eps = Epsilon parameter of DBSCAN, It is the furthest distance at which a point will pick its neighbours
#points are list : for example RT values in the list format
# eps = RT difference in minutes
def clusteringSliding(points, eps=1):
    clusters = []
    points_sorted = sorted(points)
    curr_point = points_sorted[0]
    curr_cluster = [curr_point]
    for point in points_sorted[1:]:
        if point <= curr_point + eps:
            curr_cluster.append(point)
        else:
            clusters.append(curr_cluster)
            curr_cluster = [point]
        curr_point = point
    clusters.append(curr_cluster)
    return clusters






def select_singleton_cluster(row):
    max_int_rt_dict = row.max_int_rt_dict
    if len(max_int_rt_dict.keys()) == 1:
        #found the singleton cluster
        return list(max_int_rt_dict.values())[0]
    else:
        return -1





def select_first_cluster(row, eps):
    rt_clusters = row["RT_peaks_final_eps{}".format(eps)]
    if rt_clusters != -1:
        return rt_clusters[0]
    else:
        return rt_clusters




def rt_non_tailed_multicluster(row):
    rt_known = row["final_RT_multipsm_multicluster"]
    int_rt_dict = row["max_int_rt_dict"]
    if rt_known == -1:
        max_int = np.max(list(int_rt_dict.keys()))
        rt = int_rt_dict[max_int]
        return rt
    else:
        return rt_known




def extractRT(out_table, eps):
    res_f1 = out_table.groupby("peptide_charge").agg(list)
    res_f1["nPSMs"] = res_f1.apply(lambda x: len(x.ms2_rt), axis=1)
    res_f1["RT_Clust_eps_{}".format(eps)]=res_f1.apply(lambda x: clusteringSliding(x.ms2_rt, eps=2), axis=1)
    res_f1["rt_int_dict"] = res_f1.apply(lambda x: int_rt_dict(x.ms2_rt, x.prec_intensity), axis=1)
    '''
    select singleton cluster
    1. check the length of RT_Clust_eps_2 (for eps = 2) and see if the len of list of list is 1
    2. if so extract the intensity of the list of list using rt_int_dict
    3. Assign maximum intensity_rt as the dictionary for the cluster
    4. Select the dictionary that has single key
    4. Assign that RT

    '''
    res_f1["max_int_rt_dict"] = res_f1.apply(lambda x: getMaxIntCluster(x["RT_Clust_eps_{}".format(eps)], x.rt_int_dict), axis=1)
    
    #these are best RTs and as tehre is only one option

    res_f1["weighted_rt_list"] = res_f1.apply(lambda x: weighted_average_each_cluster(x["RT_Clust_eps_{}".format(eps)], x.rt_int_dict), axis=1)
    
    #res_f1["final_RT_singleton"] = res_f1.apply(select_singleton_cluster, axis=1) #max intensity based RT
    res_f1["final_RT_singleton"] = res_f1.apply(select_singleton_cluster_wtrt, axis=1) #weighted RT
    '''
    This approach helped 98% to be resolved

    CASE1 all final_RT_singleton == -1 are multiclusters

    3 more cases of multiclusters
    CASE2 = multipeak clusters [MC] --> The clusters have multiple peaks
    CASE4 = single peak clusters/ sigleton cluster [SC]
    CASE3 = [MC]+[SC]


    #get Case 2, 3, 4
    #get multipeak cluster (Case 2)


    '''
    #use filter to remove singleton peaks for multiple occurrence
    res_f1[["RT_peaks_evaluate_eps{}".format(eps),"clusterType"]] = res_f1.apply(evalute_rt_cluster,column="RT_Clust_eps_{}".format(eps), axis=1)
    #idea is to use earlier LC profile for tailed peaks

    # res_f1["final_RT_case2"] = res_f1.apply(inferRT_Case2, clusterExplore = "Case2", column1 = "final_RT_singleton", column2 ="RT_peaks_evaluate_eps{}".format(eps) , axis=1) #max intensity
    res_f1["final_RT_case2"] = res_f1.apply(inferRT_Case2_wtrt, clusterExplore = "Case2", column1 = "final_RT_singleton", column2 ="RT_peaks_evaluate_eps{}".format(eps) , axis=1) #weighted rt


    #multiple singleton cluster
    res_f1["final_RT_case4"] = res_f1.apply(inferRT_Case4, clusterExplore = "Case4", column1 = "final_RT_case2", column2 ="RT_peaks_evaluate_eps{}".format(eps) , axis=1)
    
    #use filter to remove singleton peaks for multiple occurrence could be subcase2 and subcase4
    res_f1["subClusterTypeCase3"] = res_f1.apply(evalute_rt_cluster_case3,column="RT_peaks_evaluate_eps{}".format(eps), axis=1)
   
    # res_f1["final_RT_case3_subcase2"]=res_f1.apply(inferRT_case3_subcase2, eps = eps, axis=1) #RT based on max intensity
    res_f1["final_RT_case3_subcase2"]=res_f1.apply(inferRT_case3_subcase2_wtrt, eps = eps, axis=1) #weighted RT

    res_f1["Final_RT"]=res_f1.apply(inferRT_case3_subcase1, eps = eps, axis=1) # same as final_RT_case3_subcase1 and all others


    return res_f1







def formatRtTable2(df, runs):
    df_nPSMs = df.set_index(['key', 'run']).nPSMs.unstack().reset_index()
    df_RT = df.set_index(['key', 'run']).RT.unstack().reset_index()
#     df_RT = df.set_index(['key', 'run']).pseudoRT.unstack().reset_index()

    df_nPSMs2 = df_nPSMs.set_index("key")
    df_RT2 = df_RT.set_index("key")
    #apply runs columns to maintain the order. Unstack sort the columns based on string and 1, 10, 100 and so on we need correct order
    df_RT2 = df_RT2[runs]
    col_keys_nPSM ={}
    for val in runs:
        new_val = val+"_nPSMs"
        col_keys_nPSM[val]=new_val

    df_nPSMs3 = df_nPSMs2.rename(columns=col_keys_nPSM)
    df_nPSMs3 = df_nPSMs3[list(col_keys_nPSM.values())]
    
    res = pd.concat([df_RT2,df_nPSMs3], axis=1)
    res.reset_index(inplace=True)
    return res









def inferRT_afterSearch(psms, runs, eps=1):
    # Input
    # 1. mzXML files
    # 2. datframe from search file containing all identified PSMs
    print("  Extraction and assignment of RTs to the identified PSMs")
    print("  =======================================================")

    
    # RT extraction/assignment for each mzXML file
    key_list = []
    run_list = []
    rt_list = []
    npsm_list = []
    pseudo_rt_list = []

    ext_data_dict = {} # this dictionary has runwise dataframes 

    runName_list = []
    # ++++ raw_data_type ++++
    raw_data_type = "mzXML"
    if '.mzML' in runs[0]:
        raw_data_type = "mzML"

#     print (runs)
    for run in runs:
        
        runName = os.path.basename(run).split(".")[0]
        runName_list.append(runName)
        print("  Working now on extracting RTs from {}.\n".format(run))
        
        
        print("  RT of every identified peptide in {} is being inferred and assigned".format(runName))
        out_table = get_rt(psms, run, raw_data_type)
        # out_table.to_csv("test.out_table.txt",sep="\t",index=None)
        # out_table.to_pickle("test.pkl")

        res_f1 = extractRT(out_table, eps)

        res_f1.to_excel(runName+"_RT_extracted.xlsx")
        
        #update dictionary
        ext_data_dict[runName] = res_f1
        
        key_list.extend(list(res_f1.index))
        run_list.extend([runName]*res_f1.shape[0])
        rt_list.extend(list(res_f1.Final_RT))

        #makes a pseudo RT list for each file run .... the total run time is extracted by get_run_len
#         pseudo_rt_eachfile = np.array(list(res_f1.Final_RT))/get_run_len(run)*100 #normalizes all runs to 100 RT 
#         pseudo_rt_list.extend(pseudo_rt_eachfile)

        npsm_list.extend(list(res_f1.nPSMs))
        
        print("  Completed extracting RTs from {}.\n".format(run))

    
#     res = pd.DataFrame({"key":key_list,"run":run_list,"RT":rt_list,"pseudoRT":pseudo_rt_list,"nPSMs":npsm_list})
    res = pd.DataFrame({"key":key_list,"run":run_list,"RT":rt_list,"nPSMs":npsm_list})
    
    res = formatRtTable2(res, runName_list) #Previous formatRtTable function is replaced by formatRtTable2. This fucntion is very quick compared to previous one
    
    return ext_data_dict,res



def alignRT_aftersearch(df, runName, tol_min=1):
    
    #make a matrix for modeling
    #JDscore > 0.95
    #Type == Target
    #removes M@ peptides from modeling. this is important as they have multiple peaks 
    
    res = df[(df.JDscore > 0.95) & (df.Type == "Target") & (~df["key"].str.contains("M@"))]
    
    # Input: df = a pandas dataframe containing keys (peptide-charge pairs) and RTs over the runs (i.e., mzXML files)
    #        runs = the list of mzXML runnames in order of needed alignment
    # tol_min = tolerance of RT for deciding the consensus RT
    
    '''
    We will get 5 populations of peptides (shared <2 mins, shared >= 2 mins, reference unique, target unique, not existing in ref and target)
    For shared peptides, Use weighted averaged RT based on PSMs (ref has original RT and target has calibrated RT), record the sum PSM and SD of the RT (Pop1)
    For shared peptids Pop2, (>= 2minutes), select max psms and retain rt, if psms# is same retain RT from reference
    For reference unique peptides, keep the original reference inferred RT
    For target unique peptides, keep the calibrated RT
    If the peptide does not exist in current ref or target, the values are na
    Generate a new reference (concatenating all populations of peptides along with their aligned RT)'''
    
    print("  Alignment and calibration of RTs over {} runs".format(runName))
    print("  ==============================================\n")
    print("  Library RT is selected as the reference fraction")
    
    keys = res["key"]

    #initializing the loess function in R
    # rLoess = loess()
    # rPredict = ro.r("predict")
    
    deltaRT_recorder = []
    print ("    The reference run is now being aligned with target runs")

     
    ref = res["RT"] #this is always the reference run and it is updated
    target = res[runName] #this keeps changing with each loop based on given run list


    # Build a LOESS model and calibrate RTs
    # mod = rLoess(FloatVector(ref), FloatVector(target))  # LOESS model based on the shared peptides
    # cal_target = rPredict(mod, FloatVector(df[runName]))  # Calibration is applied to whole peptides of the current run

    '''
    #chatgpt code for extended data extrapolation using lowess
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    # Generate a sample dataset
    np.random.seed(0)
    x = np.linspace(-2, 2, 50)
    y = x**2 + np.random.normal(0, 0.5, size=50)

    # Extend the range of the independent variable
    x_extended = np.linspace(-3, 3, 50)

    # Fit a LOESS curve to the original range
    loess = sm.nonparametric.lowess(y, x, frac=0.5)

    # Plot the sample data and the LOESS curve
    plt.scatter(x, y, label='Sample Data')
    plt.plot(x_extended, np.interp(x_extended, loess[:, 0], loess[:, 1]), label='LOESS Curve', color='red')
    plt.legend()
    plt.show()

    '''
    # Fit a LOESS curve to the extended range
    loess = sm.nonparametric.lowess(target, ref, frac=0.5)
#    cal_target =  np.interp(df[runName], loess[:, 0], loess[:, 1])    
    cal_target =  np.interp(df[runName], loess[:, 1], loess[:, 0])


    #replace target runs with calibrated RT
    df["calibratedRTs"] = cal_target
    #selection of consensus RT
    delRT = df.RT - cal_target # difference between the reference run and calibrated target run

    delRT_R_T = pd.DataFrame({"key":keys, "delRT":delRT})
    deltaRT_recorder.append(delRT_R_T)
        
        
    return deltaRT_recorder


def getOrderedMzxmlList(mzxml_path, orderedFraction,raw_data_type): #orderedFraction = file that contians ordered fractions with one fraction in one row. 
    #this updates the new mzXML list based on the ordered fraction list provided
    df = pd.read_csv(orderedFraction, delimiter="\t", header=None)
    df["mzXML"] = mzxml_path+"/"+df[0]+".{}".format(raw_data_type)
    mzXML_list = list(df["mzXML"])
    return mzXML_list 



###### FUnction to find the total run lenght of mzXML file ####

def get_run_len(mzxml):
    f = open(mzxml,"r") #read the file
    line = f.readline()
    var_AA_mass = {} 
    var_AA_symbol = {} #symbol mass dictionary
    stat_AA_mass = {}
    while "<msRun scanCount=" not in line: #end reading the file if this is sen
        line = f.readline()

    #     #<msRun scanCount="35023" startTime="PT480.156S" endTime="PT5400.19S" >
    pattern = 'endTime="PT(\d+(\.\d+)?)S"'
    
    m= re.search(pattern, line)
#     print (mzxml,float(m[1])/60)
    
    return float(m[1])/60 # time in minutes



#################################SUMMARY###########################



def summary(filename,df,delRT_Col = "delRT"): # df is alignment with 2 fractions only
    
    overlapped_prec_list = []
    rt_tolerance_list = []
    percentage_prec_list = []
    
    with open(filename,"w") as f:
        f.write("Overlapped Precursor\tdelta RT (tol)\tPercentage (%) precursors\n")
        print ("Overlapped Precursor\tdelta RT (tol)\tPercentage (%) precursors")
        for x in np.arange(0.5, 10, 0.5):
            cnt = df.loc[abs(df[delRT_Col]) < x].shape[0]
            f.write("{}\t{}\t{}\n".format(cnt, x, cnt/df.shape[0]*100))
            print ("{}\t{}\t{}".format(cnt, x, cnt/df.shape[0]*100))
            overlapped_prec_list.append(cnt)
            rt_tolerance_list.append(x)
            percentage_prec_list.append(cnt/df.shape[0]*100)
    newDF = pd.DataFrame({"Overlap_Prec":overlapped_prec_list,"rt_tolerance":rt_tolerance_list,"overlap_prec_percentage":percentage_prec_list})
    
    return newDF

'''
# Suresh method infered RT file check delRT before alignment
df_pkl = "/home/spoudel1/spectral_library_manuscript/extract_RT/program_ms2_based_RT/serum_samples/all_fractions_RT.pkl"
df_suresh_infer = pd.read_pickle(df_pkl)

df_suresh_infer["delRT_before"] = df_suresh_infer.Yadav_B10 - df_suresh_infer.Yadav_B11 

#remove na Suresh inference
df_suresh_infer_noNA = df_suresh_infer.dropna()
suresh_before = summary("suresh_inference_before_alignment",df_suresh_infer_noNA,delRT_Col = "delRT_before")


'''


################### FUnctions for consensus RT selection after alignment ###################



def pop2_rt_consensus(row, runs,exp, run_n_psm):
    keys = row["key"]
    ref =  float(row[runs[0]])
    target =  float(row[runs[exp]])
    
    refPSM =  float(row[run_n_psm[0]])
    targetPSM =  float(row[run_n_psm[exp]])
    
    if (refPSM > targetPSM) | (refPSM == targetPSM):
        return pd.Series([ref, refPSM])
#         final_rt.append(ref)
#         final_psms.append(refPSM)
    else: 
        return pd.Series([target, targetPSM])

#         final_rt.append(target)
#         final_psms.append(targetPSM)
    
#     return pd.Series([final_rt, final_psms])


def weighted_average2(row,runs,exp,run_n_psm):
    ref =  float(row[runs[0]])
    target =  float(row[runs[exp]])
    
    refPSM =  float(row[run_n_psm[0]])
    targetPSM =  float(row[run_n_psm[exp]])
    
    refVal = ref*refPSM
    tarVal = target*targetPSM
    
    num = refVal+tarVal
    den = refPSM+targetPSM
    
    weightedRT = num/den
    
    return weightedRT


def weighted_average(df, rt_cols_list, rt_weights_cols_list):
    #numpy array of rt values for 2 columns rt_cols_list
    rt_values = df[rt_cols_list].values
    #numpy array of rt weights here psms for 2 columns rt_weights_cols_list

    rt_weights = df[rt_weights_cols_list].values
    #ignore nan when summing across the axis =1 (row)
    weighted_rt = np.nansum(rt_values*rt_weights, axis=1)/np.nansum(rt_weights, axis=1)
#     weighted_rt = (df[rt_cols_list].values*df[rt_weights_cols_list].values).sum(axis=1)/df[rt_weights_cols_list].values.sum(axis=1)
    return weighted_rt
