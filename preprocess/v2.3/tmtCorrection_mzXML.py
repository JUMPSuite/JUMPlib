#import sys
import pandas as pd
#import os
import re
import numpy as np
#from tmtCorrection_mzXML import *
#import pandas as pd
import pyteomics
from pyteomics import mzxml,mzml,mass
from process_ms3 import *

def all_scans_TMTcorrection(np_arr1,np_arr2,tmt, y1_tmt_Lys, y1_Arg,tol_max):
    
#     print (np_arr2)
    correctionFactor_tmt = []
    correctionFactor_y1Lys_tmt = []
    correctionFactor_y1Arg = []
    
    # if the ms scan does not have the reference ions, the mass shift will be 2*tol_max
    # mass shift for each type (tmt, y1Lys_tmt, y1Arg)
    massErrors_tmt = np.zeros((len(np_arr2),1))+2*tol_max
    massErrors_y1Lys_tmt = np.zeros((len(np_arr2),1))+2*tol_max
    massErrors_y1Arg = np.zeros((len(np_arr2),1))+2*tol_max
    i=0
    for val in np_arr2:
        scan = str(val[0])
        exp_mz_list = val[8].tolist()
        intensity_list = val[9].tolist()
        #correctionFactor = massShiftCalculator(exp_mz_list,intensity_list,tmt,tol_max)# all tmt ions are used
        correctionFactor = massShiftCalculator(exp_mz_list,intensity_list,[tmt[0]],tol_max)# only tmt 126 is used
        correctionFactor_tmt+=correctionFactor
        correctionFactor_y1tmtLys = massShiftCalculator(exp_mz_list,intensity_list,[y1_tmt_Lys],tol_max)
        correctionFactor_y1Lys_tmt+=correctionFactor_y1tmtLys
        correctionFactor_y1R = massShiftCalculator(exp_mz_list,intensity_list,[y1_Arg],tol_max)
        correctionFactor_y1Arg+=correctionFactor_y1R
        
        # mass shift for each type (tmt, y1Lys_tmt, y1Arg)
        if len(correctionFactor)>0:
            massErrors_tmt[i]=np.mean(correctionFactor)
        if len(correctionFactor_y1tmtLys)>0:
            massErrors_y1Lys_tmt[i]=np.mean(correctionFactor_y1tmtLys)
        if len(correctionFactor_y1R)>0:
            massErrors_y1Arg[i]=np.mean(correctionFactor_y1R)
        i=i+1
    
    # select the type with the largest number of ms scans
    nvalidppm = [len(np.nonzero(massErrors_tmt<=tol_max)[0]),len(np.nonzero(massErrors_y1Lys_tmt<=tol_max)[0]),len(np.nonzero(massErrors_y1Arg<=tol_max)[0])]
    ix=np.argmax(nvalidppm)
    if ix==0:# tmt
        massErrors=massErrors_tmt
    elif ix==1:# y1Lys_tmt
        massErrors=massErrors_y1Lys_tmt
    else:# y1Arg
        massErrors=massErrors_y1Arg
    
    return correctionFactor_tmt,correctionFactor_y1Lys_tmt,correctionFactor_y1Arg,massErrors

def mzFileToNumpyArr(mzFILE,raw_data_type):
    #mzXcols_old=['num','msLevel','m/z array','intensity array','retentionTime','precursorMz']
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
        
        ms1 = dfmzXML.loc[dfmzXML.msLevel==1]     #ms1 level scans
        np_arr1 = ms1.to_numpy()
        ms2 = dfmzXML.loc[dfmzXML.msLevel==2]     #ms2 level scans
        np_arr2 = ms2.to_numpy()
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
        
        ms1 = dfmzXML.loc[dfmzXML.msLevel==1]     #ms1 level scans
        np_arr1 = ms1.to_numpy()
        ms2 = dfmzXML.loc[dfmzXML.msLevel==2]     #ms2 level scans
        np_arr2 = ms2.to_numpy()
    else:
        data = {"num": [], "msLevel": [], "peaksCount": [], "retentionTime": [], "msType": [], "activationMethod": [], "precursorMz": [], "precursorCh": [], "m/z array": [], "intensity array": [], "precursorIntensity": [], "basePeakIntensity": []}
        dfmzXML = pd.DataFrame(data)
        np_arr1 = []
        np_arr2 = []
    
    return dfmzXML,np_arr1,np_arr2

def massShiftCalculator(exp_mz_list,intensity_list,standardIonsList,tol_max=15): # it will be used for both MS1 and MS2, the highest peak will be selected
    checkMinReporter = standardIonsList[0]
    checkMaxReporter = standardIonsList[-1]
    
    min_exp_mzCheck = float(checkMinReporter)-(tol_max/100)
    max_exp_mzCheck = float(checkMaxReporter)+(tol_max/100)
    
    calibrationList = []
    calibrationListInt = []
    
    for i, val in enumerate(exp_mz_list):
        if min_exp_mzCheck <= float(val) <= max_exp_mzCheck:
            calibrationList.append(float(val))
            calibrationListInt.append(intensity_list[i])
    
#     print (calibrationList)
    reporterDict = {}            
    for reporters in standardIonsList:
        for index, masses in enumerate(calibrationList):
        
            massshift = ppmCalc(float(reporters), float(masses))
            if abs(massshift) > tol_max:
                pass
            else:
                if reporters not in reporterDict.keys():
                    reporterDict[reporters] = [[massshift,calibrationListInt[index],masses]]
                else:
                    reporterDict[reporters].append([massshift,calibrationListInt[index],masses])
    
    massShiftList = []
    if len(reporterDict)>0:
        for key in reporterDict.keys():
            intVal=0
            for subVal in reporterDict[key]:
                if subVal[1] >= intVal:
                    intVal = subVal[1]
                    massShiftKeep = subVal[0]
            massShiftList.append(massShiftKeep)
    
    return massShiftList

def calibratedMass(mz, massError):
    calibMass = mz/(1+(massError/1e6))
#     calibMass = mz - ((massError*mz)/1e6)
    return calibMass

def massCorrectionFunction(exp_list, massError):
    massCorrectedList = []
    for mz in exp_list:
        calibMass = calibratedMass(mz, massError)
        massCorrectedList.append(calibMass)
    return massCorrectedList

def ppmCalc(a, b, tol=10):
    a = float(a)
    b = float(b)
  #   massError = abs(a-b)*1e6/a
    massError = (b-a)*1e6/a  #calculates ppm error of the a(theoretical) from b(observed)
    return float(massError)

def MS2MassCorrection(np_arr1,np_arr2,tmt, tol_max):
    
    tmt_tag_mass = {"TMT0" :224.1524779, "TMT2" :225.1558327,
                    "TMT6" :229.1629321, "TMT7" :229.1629321,
                   "TMT8" :229.1629321, "TMT9" :229.1629321,
                    "TMT10" :229.1629321, "TMT11" :229.1629321,
                   "TMT16":304.2071453}
    if len(tmt)==1:
        tmtReport = "TMT0"
    else:
        tmtReport = "TMT%d" % len(tmt)
    
    proton = 1.00727646677 #proton monoisotopic mass
    y1_Arg = str(mass.calculate_mass("R")+(1*proton)) #y1 ion with Arginine 
    y1_tmt_Lys = str(mass.calculate_mass("K")+(1*proton)+tmt_tag_mass[tmtReport]) #y1 ion with Lysine and TMT ion
    
    reportAllShift,lysineAllShift,arginineAllShift,massErrors = all_scans_TMTcorrection(np_arr1,np_arr2,tmt, y1_tmt_Lys, y1_Arg, tol_max)
    MS1_index=np.zeros((len(np_arr1),5)) # MS1 scan, MS1 rt, MS1 peak num, baseline, massshift
    MS2_index=np.zeros((len(np_arr2),8)) # MS1 scan,MS1 rt,MS2 scan,m/z,z,Fragtype,MS2 peak num,massshift
    i=0
    for val in np_arr1:
        MS1_index[i,0]=int(val[0])
        i=i+1
    i=0
    for val in np_arr2:
        MS2_index[i,2]=int(val[0])
        i=i+1
    
    # superListMassShift = reportAllShift+lysineAllShift+arginineAllShift
    # correctionFactor = np.mean(superListMassShift)
    pos = np.nonzero(massErrors<=tol_max)[0]# get the mass shift with valid values (i.e. ms scans have references ions)
    if len(pos)==0:
        correctionFactor = 0
    else:
        correctionFactor = np.mean(massErrors[pos])
    # print([1,len(reportAllShift),np.mean(reportAllShift)])
    # print([2,len(lysineAllShift),np.mean(lysineAllShift)])
    # print([3,len(arginineAllShift),np.mean(arginineAllShift)])
    
    return MS1_index,MS2_index,massErrors

def MS1MassCorrection(np_arr1,np_arr2, referenceMasses, tol_max):
    
    MS1_index=np.zeros((len(np_arr1),5)) # MS1 scan, MS1 rt, MS1 peak num, baseline, massshift
    MS2_index=np.zeros((len(np_arr2),8)) # MS1 scan,MS1 rt,MS2 scan,m/z,z,Fragtype,MS2 peak num,massshift
    i=0
    for val in np_arr1:
        MS1_index[i,0]=int(val[0])
        i=i+1
    i=0
    for val in np_arr2:
        MS2_index[i,2]=int(val[0])
        i=i+1
    
    massErrors = np.zeros((len(np_arr1),1))+2*tol_max
    i=0
    for val in np_arr1:
        scan = str(val[0])
        exp_mz_list = val[8].tolist()
        intensity_list = val[9].tolist()
        correctionFactor_ref = massShiftCalculator(exp_mz_list,intensity_list,referenceMasses,tol_max=15)
        if len(correctionFactor_ref)>0:
            massErrors[i]=np.mean(correctionFactor_ref)
        i=i+1
        
    return MS1_index,MS2_index,massErrors
