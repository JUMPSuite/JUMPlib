import pandas as pd
# from calcMS2Similarity import *
import re, numpy as np
import math
import os
from pyteomics import mass

#this is for dynamic intensity to SD relationship
def parseDynamicIntensityFile(file, n=3):
    dfDyn = pd.read_csv(file, delimiter ="\t")
    int_sd_dict = dict(zip(np.round(dfDyn.log10Intensity,5), n*dfDyn.SD))
    return int_sd_dict




def ppmCalc(a, b, tol=10):
    a = float(a)
    b = float(b)
  #   massError = abs(a-b)*1e6/a
    massError = (b-a)*1e6/a  #calculates ppm error of the a(theoretical) from b(observed)
    return float(massError)



def ms2ToDf_spec(ms2File, sim_mass=0.0):
    g = open(ms2File,"r")
    lines = g.readlines()
    scan_list = []
    charge_list = []
    MH_list = []
    precursorIon_list = []
    L_ID_list = []
    L_peptide_list = []
    L_Protein_list = []
    RT_list = []
    ms2_mz = []
    ms2_int = []

    mz_list = []

    for line in lines:
        if "S\t" in line:
            if len(mz_list) >= 1:
                ms2_mz.append(mz_list)
                ms2_int.append(int_list)
            mz_list = []
            int_list = []
    #         print ("spectrum found")
            temp_line = line.strip().split("\t")
            scan = int(temp_line[1])
            scan_list.append(scan)
            precursorIon = float(temp_line[-1])+sim_mass
            precursorIon_list.append(precursorIon)
        elif "Z\t" in line:
    #         print ("Charge found")
            temp_line = line.strip().split("\t")
            charge = int(temp_line[1])
            charge_list.append(charge)

            MH = float(temp_line[-1])
            MH_list.append(MH)

        elif "L\tID_with_Modification" in line:
    #         print ("peptide ID found")
            temp_line = line.strip().split("\t")
            L_ID = temp_line[2]
            L_peptide = temp_line[3]
            L_ID_list.append(L_ID)
            L_peptide_list.append(L_peptide)
        elif "L\tProtein" in line:
            temp_line = line.strip().split("\t")
            L_Protein = temp_line[-1]
            L_Protein_list.append(L_Protein)
        
        elif "L\tRT" in line:
    #         print ("RT found")
            temp_line = line.strip().split("\t")
            RT = float(temp_line[2])
            RT_list.append(RT)
        elif re.match(r"^\d+.*$",line):
    #         print ("ms2 FOund")
            #temp_line = line.strip().split("\t")
            temp_line = re.split("\t| ",line) #to adapt to Zuo-Fei's ms2 pattern
            mz_list.append(float(temp_line[0]))
            int_list.append(float(temp_line[1]))

    ms2_mz.append(mz_list)
    ms2_int.append(int_list)

    dict1 = {"scan":scan_list,"charge":charge_list,"[M+H]+":MH_list,
                "prec_MZ":precursorIon_list,"L_ID":L_ID_list,"L_peptide":L_peptide_list,
                "L_protein":L_Protein_list,"RT":RT_list,"m/z":ms2_mz,"intensity":ms2_int}

    dict2 = {}
    for key in dict1.keys():
        if len(dict1[key]) != 0:
            dict2[key] = dict1[key]

    ms2Df = pd.DataFrame.from_dict(dict2)
    
    return ms2Df



#This normalization method is not being used now
def normalizeIntensity_log2(df):
    norm_intensity = []
    mz_cols = list(df.columns) #for indexing columns
    np_arr = df.to_numpy() #numpy array conversion of dataframe
    for row in np_arr: #each matched library checked for top ions according to parameters above
        intensity = row[mz_cols.index("intensity")] #intensity list (ms2)
        intensity_log2 = np.sqrt(intensity)
        maxval = np.max(intensity_log2)
        normTest = intensity_log2/maxval*100
        norm_intensity.append(normTest)
    df["normalized_intensity"] = norm_intensity

#This normalization method is used to convert intensity to 100
#it takes the parameter window number 
def normalizeIntensity(df, window_number):
    norm_intensity = []
    mz_cols = list(df.columns) #for indexing columns
    np_arr = df.to_numpy() #numpy array conversion of dataframe
    for row in np_arr: #each matched library checked for top ions according to parameters above
        mz = row[mz_cols.index("m/z")]
        intensity = row[mz_cols.index("intensity")] #intensity list (ms2)
        #windowLen = math.ceil(len(mz)/window_number)
        norm_intensity_win = []


        backupWin = window_number
        
        while (backupWin != 1) and (len(mz) < 10*backupWin):
            #print ("Total ions in the spectrum is too few for window ", backupWin, "attempting ", backupWin -1," windows")
            backupWin = backupWin -1
            if backupWin == 1:
                break
        
        windowLen = math.ceil(len(mz)/backupWin) 
        for x in  range(0,backupWin):
                
            startVal = x*windowLen
            endVal=windowLen+startVal
            if endVal > len(mz):
                endVal == len(mz)
            intensityWindow = intensity[startVal:endVal]
#             print (intensityWindow)
    #         print (len(intensityWindow))
            maxval = np.max(intensityWindow)
            normTest = intensityWindow/maxval*100
        #         normTestSqrt = np.sqrt(normTest)
            norm_intensity_win+=list(normTest)
#             print (len(norm_intensity_win))
        norm_intensity.append(norm_intensity_win)
  #      else:
   #         maxval = np.max(intensity)
    #        normTest = intensity/maxval*100
    #         normTestSqrt = np.sqrt(normTest)
     #       norm_intensity.append(normTest)
    df["normalized_intensity"] = norm_intensity


#This normalization method is used to convert intensity to 100
def normalizeIntensitySingleWin(df):
    norm_intensity = []
    mz_cols = list(df.columns) #for indexing columns
    np_arr = df.to_numpy() #numpy array conversion of dataframe
    for row in np_arr: #each matched library checked for top ions according to parameters above
        intensity = row[mz_cols.index("intensity")] #intensity list (ms2)
        maxval = np.max(intensity)
        normTest = intensity/maxval*100
#         normTestSqrt = np.sqrt(normTest)
        norm_intensity.append(normTest)
    df["normalized_intensity"] = norm_intensity



#This log10 intensity for dynamic mass tolerance
def logTransformMS2Intensity(df):
    log10_intensity = []
    mz_cols = list(df.columns) #for indexing columns
    np_arr = df.to_numpy() #numpy array conversion of dataframe
    for row in np_arr: #each matched library checked for top ions according to parameters above
        intensity = row[mz_cols.index("intensity")] #intensity list (ms2)
        logval = np.log10(intensity)
        
        log10_intensity.append(logval)
    df["log10_intensity"] = log10_intensity




#tol = 10 is the fragment ion tolerance ... this will be dynamic
#matched_library_DF is the matched precursor for library based on precursor and charge match
#exp_mz_list is the experimental spectrum mz list

#this function cleans the exp mz ions by removing reporter ions and its repective intensity
def generateMZ_Int_NoTMT(exp_mz_list, exp_int_list, tmt, tol):

    tmtFloat = [float(j) for j in tmt] #convert to float
    maxTMT = np.max(tmtFloat) #maximum tmt reporter ion
    cutoffMass = maxTMT + (float(maxTMT)/1000000*tol)  #add tolerance to remove maximum tmt reporter ion, this can be larger too
    
    new_exp_mz = []
    new_exp_int = []

    for i,x in enumerate(exp_mz_list):
        if x > cutoffMass:
            new_exp_mz.append(x)
            new_exp_int.append(exp_int_list[i])

    return new_exp_mz,new_exp_int


def checkTopLibraryIons(exp_mz_list, matched_library_DF, min_top_ions, top_ion=3, tol=10): #top_ion is the minimum no of ions (intensity ranked top) that should be present in experimental mz 
    checkIons = top_ion -1 #total number of ions in the library    
 
    final_count = [] #this checks if top_ion are present in exp_mz_list
    
    #this is to speed up the scanning process as numpy array are fastest
    mz_cols = list(matched_library_DF.columns) #for indexing columns
    np_arr = matched_library_DF.to_numpy() #numpy array conversion of dataframe
    for row in np_arr: #each matched library checked for top ions according to parameters above
        mz = row[mz_cols.index("m/z")] #mz list (ms2)
        intensity = row[mz_cols.index("intensity")] #intensity list (ms2)
        scan = str(row[mz_cols.index("scan")]) #scan 
       
        ind = np.argsort([-i for i in intensity]) #sorting by descending order intensity
        

        top_ion_lib = [] #empty list for top ion collection
        for i in ind[0:top_ion]: #iterating over top intensity index
            top_ion_lib.append(mz[i]) #storing top intensity asociated mz


        cnt = 0
        for ion in top_ion_lib: #looping over top_ions to see if they are present in experimental mz
            max_mz_check = float(ion)+(float(ion)/1000000*tol) #checking maximum mz ion within the fragment tolerance
            min_mz_Check = float(ion)-(float(ion)/1000000*tol) #checking minimum mz ion within fragment tolerance
            
            for mzIon in exp_mz_list:
                if (min_mz_Check <= mzIon <= max_mz_check):
#                     print (mzIon)
                    cnt+=1
                    break
#         print (cnt)
            
                
        final_count.append(cnt)
        # print ("\n\n    The total matched count is ",final_count)


    matched_lib_DF_top = matched_library_DF.copy(deep=False)
    matched_lib_DF_top["topIonsExist"] = final_count
    #matched_lib_DF_top2 = matched_lib_DF_top.loc[matched_lib_DF_top.topIonsExist >= checkIons]    
    #this was too stringent as the matched scan should have all top ions present to be balid

    #This ensures that we have at least 1 out of user defined top ions present in the library
    matched_lib_DF_top2 = matched_lib_DF_top.loc[matched_lib_DF_top.topIonsExist >= min_top_ions]
    

    return matched_lib_DF_top2


# In[50]:


def massCutoffArginine(exp_mz_list, exp_int_list):
    cutoffMass = 175 #Arginine mass ... this will ensure no TMT ions are present
   
    new_exp_mz = []
    new_exp_int = []

    for i,x in enumerate(exp_mz_list):
        if x > cutoffMass:
            new_exp_mz.append(x)
            new_exp_int.append(exp_int_list[i])

    return new_exp_mz,new_exp_int

#this function is used to bin 100 mz bins
#this function picks up top 10 ions from each bin and remake the list and intensity (normalized 100)


def binning_mz_100(exp_mz_list, exp_int_list, top_ions, binsize):#top_ions = 10 #take highest top 10 ions using intensity to rank
    
    mz_list, intensity_list = massCutoffArginine(exp_mz_list, exp_int_list)

    top_mz_list = []
    top_intensity_list = []
    
    maxv = np.max(mz_list)+binsize
    minv = np.min(mz_list)-binsize
    bins = np.arange(minv, maxv, binsize) # fixed bin width 100 mz
#     print ("max Val = ", maxv)
#     print ("min Val = ", minv)
    
    start_val = 0
    for x in range(1,len(bins)):
        sub_list_mz = [mzval for mzval in mz_list if bins[x-1] <= mzval < bins[x]] #subset mz list w
        if len(sub_list_mz) >=1: #some bins may not have any ions so we need to check the bin's element
            index_int = mz_list.index(sub_list_mz[-1]) #index for last value of sub list mz
            sub_list_int = intensity_list[start_val:index_int] #extract intensity until last value of sub list using intensity list (start and end index)
            start_val = index_int+1 #update start value for next round

            ind = np.argsort([-i for i in sub_list_int]) #sorting by descending order intensity

            top_ion_lib = [] #empty list for top ion collection
            top_int_lib = [] #empty list for top intensity collection
            for i in ind[0:top_ions]: #iterating over top intensity index
                top_ion_lib.append(sub_list_mz[i]) #storing top intensity asociated mz
                top_int_lib.append(sub_list_int[i])
            top_mz_list+=top_ion_lib  #append top 6 ions in the bin
            top_intensity_list+=top_int_lib #append top 6 ions intensity in the bin
    #     print (top_mz_list)
    #     print (top_intensity_list)
    
    #the intensity used are throughout normalized for the dot product calculation
    simplifiedIonsDict = {"mz":top_mz_list,"intensity":top_intensity_list} #this dictionary stores mz and intensity after selecting top ions (example 10) after binning into 100 mz window
    
    return simplifiedIonsDict


# In[51]:


def tidy_split(df, column, sep='|', keep=False):
    """
    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.

    Params
    ------
    df : pandas.DataFrame
        dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `df`.
    """
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy(deep=False)
    new_df[column] = new_values
    return new_df


# In[52]:


#program to rank psms based on dot product
def rankMatchedPSMS(row):
    
    rankedPSMS = [] #update this with new value Rank 1 or 2 etc
    
    matchedINFO = row.simMS2 #get score column information
    allMatchesList = matchedINFO.split(",") #split them based on ,
    #L_ID+";"+str(normalized_dp)+";"+str(RT)+";"+L_peptide
    L_ID = [] #Library ID
    DP = [] #dot product
    RT = [] #RT information
    L_PEP = [] #Peptide information
    charge = [] #preadded in peptide
    precMZ = [] #preadded in peptide
    delCN = [] # added newly 070924
    match_num = [] # added newly 070924
    total_num = [] # added newly 070924
    for match in allMatchesList:
        all_vals = match.split(";")
        L_ID.append(all_vals[0]) #make sure the index order is accurate 0=L_ID, 1= DP, 2 = RT, 3 = L_PEP
        DP.append(float(all_vals[1])) #change to float to sort and later change to string after sorting
        RT.append(all_vals[2])
        L_PEP.append(all_vals[3])
        charge.append(all_vals[4])
        precMZ.append(all_vals[5])
        delCN.append(all_vals[6])
        match_num.append(all_vals[7])
        total_num.append(all_vals[8])

        # if all_vals[3] == "Decoy":
        #     charge.append(str(np.nan))
        #     precMZ.append(str(np.nan))
        # else:
        #     charge.append(all_vals[4])
        #     precMZ.append(all_vals[5])
        
    ind = np.argsort([-i for i in DP])
    
    ranked_psms = [] #empty list updating ranked psms
    for val,i in enumerate(ind): #iterating over top score index #val is index of index
        #addVal = L_ID[i]+";"+str(DP[i])+";"+RT[i]+";"+L_PEP[i]+";"+charge[i]+";"+precMZ[i]+";Rank"+str(val+1) #add Rank information starting from 1
        addVal = f"{L_ID[i]};{DP[i]};{RT[i]};{L_PEP[i]};{charge[i]};{precMZ[i]};Rank{val+1};{delCN[i]};{match_num[i]};{total_num[i]}"
        rankedPSMS.append(addVal)
    
    return ",".join(rankedPSMS) #join the list with 

        
#this is required as our advanced mass calibration expects the peptide to have flanking amino acid
#just try to mimic that 
def makeJUMP_likePeptide(row,sta_AA,jump_mod_dict):
    peptide = row.Peptide_ID
    mod_list = re.findall("\d+.\d+",peptide)
    peptide=peptide.replace("+","").replace("(","").replace(")","")
    for x in mod_list:
        if float(x) in sta_AA.values():
            peptide=peptide.replace(x,"")
        if float(x) in jump_mod_dict.values():
            for key, mods in jump_mod_dict.items():
                peptide=peptide.replace(x,key)
    return "."+peptide+"."


#this function extracts dynamic and static modifications using pepxml files and stores them as the dictionary
def getDynStatModsInfoPepXml(pepxml):
    f = open(pepxml,"r") #read the file
    line = f.readline()
    var_AA_mass = {} 
    var_AA_symbol = {} #symbol mass dictionary
    stat_AA_mass = {}
    while "<spectrum_query" not in line: #end reading the file if this is sen
    # look for aminocaid modification keyword so that the modification infomration can be parsed
        if "<aminoacid_modification" in line: #found modification information
            if "symbol=" in line.strip(): #only dynamic modification has symbols as static are fixed

                #check the patter
                pattern = '<aminoacid_modification aminoacid="(\w)" massdiff="(\d+\.\d+)" mass="\d+\.\d+" variable="(\w)" symbol="(.+?)"/>'

                #match for the patter in the line and store them as the tuples there are 4 matches () this is changed to list with list function
                mods_Description=list(re.match(pattern, line.strip()).group(1,2,3,4))

                modAA = mods_Description[0] #first element in list is amino acid 
                varMass = float(mods_Description[1]) #second element in list is mass and it is converted to float
                variable = mods_Description[2] #third element in the list is determination of variable of static modification "Y" is variable and "N" is static
                symbol = mods_Description[3] #Symbol. this is used in the dictionary
                valAddKey(var_AA_mass,  varMass, modAA)
    #             valAddKey(var_AA_symbol, symbol, varMass)
                var_AA_symbol[symbol] = varMass #symbol as key and mass as values in the dictionary
                line = f.readline()
            else:
                #this is for static modification so there is no symbol hence we have only three values in the list
                pattern = '<aminoacid_modification aminoacid="(\w)" massdiff="(\d+\.\d+)" mass="\d+\.\d+" variable="(\w)"/>'
                mods_Description=list(re.match(pattern, line.strip()).group(1,2,3))
                modAA = mods_Description[0]
                varMass = float(mods_Description[1])
                variable = mods_Description[2]
                stat_AA_mass[modAA] = varMass
    #             valAddKey(stat_AA_mass, modAA, varMass)
                line = f.readline()

        elif "<terminal_modification terminus" in line: #This is for terminal modification such as N-term or C-term
            if "symbol=" in line.strip():
                pattern = '<terminal_modification terminus="(\w)" massdiff="(\d+\.\d+)" mass="\d+\.\d+" variable="(\w)".+symbol="(.+?)"/>'
                mods_Description=list(re.match(pattern, line.strip()).group(1,2,3,4))

                modAA = mods_Description[0].lower()
                varMass = float(mods_Description[1])
                variable = mods_Description[2]
                symbol = mods_Description[3]
                valAddKey(var_AA_mass,  varMass, modAA)
    #             valAddKey(var_AA_symbol, symbol, varMass)
                var_AA_symbol[symbol] = varMass
                line = f.readline()
            else:
                pattern = '<terminal_modification terminus="(\w)" massdiff="(\d+\.\d+)" mass="\d+\.\d+" variable="(\w)".+/>'
                mods_Description=list(re.match(pattern, line.strip()).group(1,2,3))
                modAA = mods_Description[0].lower()
                varMass = float(mods_Description[1])
                variable = mods_Description[2]
                stat_AA_mass[modAA] = varMass
    #             valAddKey(stat_AA_mass, modAA, varMass)
                line = f.readline()
        else:
            line = f.readline()
    return var_AA_mass,var_AA_symbol, stat_AA_mass

def valAddKey(dict1, key, val):
    if key not in dict1.keys():
        dict1[key] = [val]
    else:
        dict1[key].append(val)
    return dict1


def calcNeutralMass(precursorMz,z):
    proton = mass.calculate_mass(formula="H+")
    # neutralMass = z*(precursorMz - proton)
    # return neutralMass

    massNeutral = (precursorMz*z) - ((z-1)*proton)
    return massNeutral
