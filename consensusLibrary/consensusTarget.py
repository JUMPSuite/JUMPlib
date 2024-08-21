import pyteomics as pyteomics
from pyteomics import mzxml
from pyteomics import mass
import pandas as pd
import os, sys, glob, re
from datetime import datetime
import collections
from idtxtMs2ModsFunctions import *
from logFunctions import *
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

def cosensusPSMTable(idtxtFile, psmsDict, precDict, jump_modAA_dict, jump_mod_dict, sta_AA, unimod_mod_infor, mzRT_df,specLibFolder, pepToProtDict, spectrumList = None):

    write_log ("  Constructing a consensus table for unique psms\n")

    final_mods_unimod_Details = unimodModsDict(jump_modAA_dict,sta_AA,unimod_mod_infor)
    idtxtdf_1 = pd.read_csv(idtxtFile, delimiter=";", skiprows=return_skiprows(idtxtFile, ";", "Peptide"))
    idtxtdf_1["spectrum"] = idtxtdf_1.apply(createOutfile, df=idtxtdf_1, axis=1)
    
    #this step performs the filtering of spectrum with Dscore < DscoreCutoff
    if spectrumList != None:
        idtxtdf = idtxtdf_1.loc[idtxtdf_1.spectrum.isin(spectrumList)]
    else:
        idtxtdf = idtxtdf_1.copy()

    idtxtdf[["exp","scan","charge"]] = idtxtdf["spectrum"].str.split(".",expand=True)
    

    idtxtdf["Peptides"] = idtxtdf["Peptide"]

    #merge RT information here using the RT dictionary
    
    idtxtdf["RT"] = (idtxtdf.Peptides+"_"+idtxtdf.charge).map(mzRT_df)

    write_log ("Following new information are being added for consensus table")
    write_log ("1. RT information\n2. plain peptide (sequence only)\n3. Modifications")
    write_log ("4. Modification position \n5. Peptide sequence along with delta mass in parentheses")
    write_log ("6. mz and intensity pairs for each MS2 scan\n7. Precursor mz information")
    write_log ("8. PSM#\n9. Library ID\n")    
    

    #For modification or PTMs information on the consensus library
    idtxtdf[["plain_peptide","modifications"]] = idtxtdf.apply(computeModifications, jump_mod_dict=jump_mod_dict,sta_AA=sta_AA,axis=1)
    idtxtdf["massPosDict"] = idtxtdf.modifications.apply(lambda x: spectrumToDict(x))
    idtxtdf["mixedDictPtmPosDelM"] = idtxtdf.massPosDict.apply(lambda x: mixedDictPtmPosDelM(x,unimod_mod_infor,final_mods_unimod_Details))
    idtxtdf["PeptideSeqWithRealDelMass"] = idtxtdf.apply(lambda x: addModValInPepSeq(x.plain_peptide,x.massPosDict), axis=1)
    idtxtdf["modsForReport"] = idtxtdf.apply(lambda x: modsForReport(x.modifications,x.plain_peptide), axis=1)

    idtxtdf["mz_int_pairs"] = idtxtdf.spectrum.map(psmsDict)
    idtxtdf["precursorMZ"] = idtxtdf.spectrum.map(precDict)


    #save all peptide and protein as dictionary with repetative protein as list of values
    #spectrum_ProtDict = {k: g["Protein"].tolist() for k,g in idtxtdf.groupby("spectrum")}
    spectrum_ProtDict = {k: g["Protein"].tolist() for k,g in idtxtdf.groupby("PeptideSeqWithRealDelMass")}

    
    idtxtdf.drop_duplicates(subset="spectrum", inplace=True, keep="first")
    
    write_log ("Total PSMS in all fractions = ",idtxtdf.shape[0])


    #applies QC program to get more confident library if ms2 matches < 3 remove all and if ms2 matches < 4 and Jscore < 20 throw
    write_log ("\nAll the PSMs with JScore < 30 are removed from our further analysis to increase the robustness of the library\n")
    
    idtxtdf1 = idtxtdf.loc[idtxtdf.XCorr.astype("float") > 30]

    idtxtdf2 = idtxtdf1.sort_values(by=["XCorr"], ascending = False) #sort by xcorr to select top 10 psms later
    

    idtxtdf2["L_ID"] = idtxtdf2.PeptideSeqWithRealDelMass+";"+idtxtdf2.charge+";"+idtxtdf2.precursorMZ
    
    

    #count the number of psms
    idtxtdf2['count'] = idtxtdf2.groupby('L_ID')['L_ID'].transform('count')
    #make a dictionary of L_ID and psms#
    dict_id = dict(zip(idtxtdf2['L_ID'] , idtxtdf2['count']))
    
    requiredCols = ['Peptides','mz_int_pairs', 'precursorMZ', 'mixedDictPtmPosDelM','PeptideSeqWithRealDelMass','L_ID','XCorr','spectrum','RT','Protein']
    idtxtConsensus = idtxtdf2[requiredCols]
    
    # idtxtConsensus[["Keep/Throw","xcorr_best"]] = idtxtConsensus.apply(QC_keep_throw_spectrum, axis=1)
    # #create a dataframe that keeps only QC passed psms
    # idtxtConsensus1 = idtxtConsensus.loc[newDF["Keep/Throw"] == "Keep"]

    idtxtConsensus2=idtxtConsensus.groupby('L_ID', as_index=False).agg(list)

    #merge PSMs count column to the respective IDs
    idtxtConsensus2["PSM#"] = idtxtConsensus2.L_ID.map(dict_id)

    write_log (  "Saving the Library ID and Protein accession to a file ")
#     protDF = idtxtConsensus2.copy()[['PeptideSeqWithRealDelMass']]
    idtxtConsensus2["Peptide"] = idtxtConsensus2.apply(lambda x: x.PeptideSeqWithRealDelMass[0], axis=1)
    idtxtConsensus2["JUMP_Peptide"] = idtxtConsensus2.apply(lambda x: x.Peptides[0], axis=1)
    # selectOneProtein(idtxtConsensus2, pepToProtDict)

    idtxtConsensus2["ProteinAccession"] = idtxtConsensus2.Peptide.map(pepToProtDict)
    
    write_log (  "Done ...\n")

    return idtxtConsensus2


def mapProteinToNRmatrix(df, protDict):
    df["Protein_list"] = df.Peptide.map(protDict)
    df["Protein_accession"] = df.apply(lambda x: ",".join(list(set(x.Protein_list))), axis=1)


def psmConsolidate(df, topPsmCnt=5): #topPsmCnt is a new parameter with default = 5 but previously we used fixed 10
    write_log ("  psms consolidation begin for top {} PSMs".format(topPsmCnt))

    mz_cols = list(df.columns)
    xcorr_list = []
    spectrum_list = []
    L_ID_list = []
    precMZ_list = []
    mzIntPairs_list = []
    mixedDictPtmPosDelM_list = []
    RT_list = []
    psm_cnt_list = [] #count total number of psms for the peptide
    uniq_prot_list = []
    jump_peptide_list = []

    for row in df.itertuples():
        xcorr = row[mz_cols.index("XCorr")+1]
        spectrum = row[mz_cols.index("spectrum")+1]
        L_ID = row[mz_cols.index("L_ID")+1]
        mz_int_pairs = row[mz_cols.index("mz_int_pairs")+1]
        precursorMZ = row[mz_cols.index("precursorMZ")+1]
        mixedDictPtmPosDelM = row[mz_cols.index("mixedDictPtmPosDelM")+1]
        RT = row[mz_cols.index("RT")+1]
        psmCnt = row[mz_cols.index("PSM#")+1]
        protein = row[mz_cols.index("ProteinAccession")+1]
        jump_peptide = row[mz_cols.index("JUMP_Peptide")+1]


        xcorr_list.append(xcorr)
        spectrum_list.append(spectrum)
        L_ID_list.append(L_ID)
        precMZ_list.append(precursorMZ)
        mzIntPairs_list.append(mz_int_pairs)
        mixedDictPtmPosDelM_list.append(mixedDictPtmPosDelM)
        RT_list.append(RT)
        psm_cnt_list.append(int(psmCnt))
        uniq_prot_list.append(protein)
        jump_peptide_list.append(jump_peptide)
    
    xcorr_list_top10 = []
    spectrum_list_top10 = []
    L_ID_list_top10 = L_ID_list #since this is only one ID and it is grouped by this value
    precMZ_list_top10 = []
    mzIntPairs_list_top10 = []
    mixedDictPtmPosDelM_list_top10 = []
    RT_list_top10 = []
    
    for x in range(0,len(xcorr_list)):
        xcorr_list_new = xcorr_list[x][0:int(topPsmCnt)] #top 10 psms .. now changed to a parameter with topPSMcount
        xcorr_list_top10.append(xcorr_list_new)
        
        spectrum_list_new = spectrum_list[x][0:int(topPsmCnt)] #top 10 psms.. now changed to a parameter with topPSMcount
        spectrum_list_top10.append(spectrum_list_new)
        
        L_ID_list_new = L_ID_list[x][0:int(topPsmCnt)]
        L_ID_list_top10.append(L_ID_list_new)
        
        precMZ_list_new = precMZ_list[x][0:int(topPsmCnt)]
        precMZ_list_top10.append(precMZ_list_new)
        
        mzIntPairs_list_new = mzIntPairs_list[x][0:int(topPsmCnt)]
        mzIntPairs_list_top10.append(mzIntPairs_list_new)
        
        mixedDictPtmPosDelM_list_new = mixedDictPtmPosDelM_list[x][0:int(topPsmCnt)]
        mixedDictPtmPosDelM_list_top10.append(mixedDictPtmPosDelM_list_new)

        RT_list_new = RT_list[x][0:int(topPsmCnt)]
        RT_list_top10.append(RT_list_new)


        
    newDF = pd.DataFrame(list(zip(mzIntPairs_list_top10,
                                  precMZ_list_top10,
                                  mixedDictPtmPosDelM_list_top10,
                                  L_ID_list_top10,
                                  xcorr_list_top10,
                                  spectrum_list_top10,
                                  RT_list_top10,
                                  psm_cnt_list,
                                  uniq_prot_list,
                                  jump_peptide_list)),
                                  columns = ['mz_int_pairs', 'precursorMZ','mixedDictPtmPosDelM', 'L_ID','XCorr','spectrum','RT_list','PSM#','Protein','Jump_Peptide'])
    
    
###Once we figure out RT alignment we can add that section here
#Currently since it is one file and all RT are same so I am just selecting first index of list
    newDF["RT"] = newDF.RT_list.apply(lambda x: x[0])
    #define charge state too
    newDF["charge"] = newDF.L_ID.str.split(";", expand=True)[1]
    
    write_log ("  Top {} psms from all fractions are now consolidated\n".format(topPsmCnt))

    return newDF




def createMS2EachPSMS_L_ID(df_all,specLibFolder, libraryNotes, libtypename):
    #some of the psms in ID.txt are not present in id_all_pep.txt which is used to make ppml files
    #to avoid such problems, we remove all the entries that do not have Protein columns

    df = df_all.dropna(subset=["Protein"])
    nullDF = df_all.loc[df_all.Protein.isnull()]
    nullDF.to_csv(specLibFolder+"/intermediate/missingIDs.txt", sep="\t", index=None)

    write_log ("  Total entries present in ID.txt but not present in id_all_pep.txt file = {} out of {} entries".format((df_all.shape[0]-df.shape[0]), df_all.shape[0]))

    mz_cols = list(df.columns)
    proton = 1.00727646677
    write_log ("  Generating .ms2 files\n")
    now = datetime.now()
    #write_log (prev_mod)
    write_log("  now =", now)
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%m/%d/%Y %H:%M %p")
    year = now.strftime("%Y")
    date_time = dt_string.split() #date is 0 index, time is 1 index, AM/PM is 2 index
    
    header_ms2 = "H\tCreationDate\t"+dt_string+"\nH\tExtractor\tMakeMS2\nH\tExtractorVersion\t1.0\nH\tComments\tMakeMS2 written by Suresh Poudel, "+year+"\nH\tExtractorOptions\tMS2/MS1\n"


    new_ms2_file = specLibFolder+"/intermediate/jumplib_human_{}_target.splib".format(libtypename) #modified the extension name to ms2pep
    write_log ("  Target IDs spectral library is being created", new_ms2_file)
    
    with open(new_ms2_file,"w") as new_ms2:
        new_ms2.write(header_ms2)
    
            
        #idea is to convert two list to one dictionary and keep updating dictionary as it sees another dictionary
        # keys = ['a', 'b', 'c']
        # values = [1, 2, 3]
        # dictionary = dict(zip(keys, values))
        #https://stackoverflow.com/questions/209840/how-do-i-convert-two-lists-into-a-dictionary
        #Update one dictionary from another https://stackoverflow.com/questions/10461531/merge-and-sum-of-two-dictionaries
        #x = {'both1': 1, 'both2': 2, 'only_x': 100}
        #y = {'both1': 10, 'both2': 20, 'only_y': 200}
        #print {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
        
        #making list for the decoy library generation
        scan_list = [] #scan list same as L id number to test second search
        precursor_mz_list = [] #list of prec mz
        charge_list = [] #list of charge
        pep_id_list = [] #list of peptide ID 
        ms2_mz_int_pair_list = [] #list of list of ms2 mz and int dictionary
#         ms2_int_list = [] #list of list of intensity
        jscore_list = []
        ms2Count_list = []
        peptide_seq_mod_list = []
        xcorr_list = []
        RT_list = []
        psm_cnt_list = []
        spectrum_list = []
        prot_list = []
        pep_seq_list = []
        jump_peptide_list = []

        id_no = 1

        # print ("The protein are as follows")

        for row in df.itertuples():
            mzIntDict = {}
            xcorr = row[mz_cols.index("XCorr")+1]
            xcorr_str = ['{:.2f}'.format(x) for x in xcorr] #convert float xcorr to two decimal string
            spectrum = ";".join(row[mz_cols.index("spectrum")+1])
            L_ID = row[mz_cols.index("L_ID")+1]
            mz_int_pairs = row[mz_cols.index("mz_int_pairs")+1]
            precursorMZ = float(row[mz_cols.index("precursorMZ")+1][0]) #because it is same precursor ions so we can take one element
            mixedDictPtmPosDelM = row[mz_cols.index("mixedDictPtmPosDelM")+1][0]#because it is same modifications so we can take one element
            xcorr_best = xcorr_str[0]
            RT = row[mz_cols.index("RT")+1]
            psm_cnt = row[mz_cols.index("PSM#")+1]
            protein = row[mz_cols.index("Protein")+1]
            jump_peptide = row[mz_cols.index("Jump_Peptide")+1]
            # print (jump_peptide)
            # print (L_ID+"\t"+protein)

            charge = int(row[mz_cols.index("charge")+1])
            spectrums = row[mz_cols.index("spectrum")+1]
            massNeutrallong = (precursorMZ*charge) - ((charge-1)*proton)
            massNeutral = np.round(massNeutrallong, 7)
            id_no_final = "p"+str(id_no).zfill(7)
            new_ms2.write("S\t"+str(id_no)+"\t"+str(id_no)+"\t"+str(precursorMZ)+"\n") #this is for fake scan for second search with comet
            new_ms2.write("Z\t"+str(charge)+"\t"+str(massNeutral)+"\n")
            
            new_ms2.write("L\tJUMP_peptide\t"+jump_peptide+"\n")
            new_ms2.write("L\tID_with_Modification\t"+id_no_final+"\t"+L_ID+"\n")
            new_ms2.write("L\tProtein\tRepresentative\t"+protein+"\n")
            mixedDictPtmPosDelMSorted = sortDictStrKeyToIntAndBackToStr(mixedDictPtmPosDelM)
            for posKey,ptmInfo in mixedDictPtmPosDelMSorted.items():
                new_ms2.write("L\tMod_Pos\t"+posKey+"\t"+ptmInfo+"\n")
            new_ms2.write("L\tSpectrum\t"+spectrum+"\n")
            new_ms2.write("L\tJScore\t"+";".join(xcorr_str)+"\n")
            new_ms2.write("L\tMH+\t"+str(massNeutral)+"\n")
            new_ms2.write("L\tPrecursor\tz="+str(charge)+"\t"+str(precursorMZ)+"\n")
            new_ms2.write("L\tRT\t"+str(np.round(RT,3))+"\n")
            new_ms2.write("L\tPSM#\t"+str(psm_cnt)+"\n")
            new_ms2.write("L\tLibraryNotes"+"\t"+libraryNotes+"\n")
            new_ms2.write("L\tTotalBatches\t1\n")
            new_ms2.write("L\tPeptideLibraryCount\t1\n")

            mz_list_new = []
            int_list_new = []
            
            
            for x in range(0,len(mz_int_pairs)):
                each_pair = mz_int_pairs[x]
        #         print (each_pair[0]) #mz value
        #         print (each_pair[1]) #intensity value
                for index,val in enumerate(each_pair[0]):
                    mz_list_new.append(float(val))
                    int_list_new.append(int(each_pair[1][index]))

                dictionary = dict(zip(mz_list_new, int_list_new)) #make dictionary using mz as keys and intensity as values
                mzIntDict = {k: mzIntDict.get(k, 0) + dictionary.get(k, 0) for k in set(mzIntDict) | set(dictionary)}
            
#             print ("JScore = ",";".join(xcorr_str), "total ms2 spectra = ",len(mzIntDict.keys()))
            
    
            #this part updates all the values for decoy library. We can add other information too if required
            scan_list.append(str(id_no))
            precursor_mz_list.append(precursorMZ)
            charge_list.append(charge)
            pep_seq_list.append(L_ID)
            pep_id_list.append(id_no_final)
            peptide_seq_mod_list.append(L_ID)
            prot_list.append(protein)

#             ms2_int_list.append(int_list_new)
            jscore_list.append(";".join(xcorr_str))
            ms2Count_list.append(len(mzIntDict.keys()))
            xcorr_list.append(xcorr_best)
            RT_list.append(RT)
            spectrum_list.append(spectrums)
            id_no+=1  ###updates ID number
            
            final_dict = collections.OrderedDict(sorted(mzIntDict.items()))
            for key,value in final_dict.items():
#                 print (key,"\t",value)
                new_ms2.write(str(key)+"\t"+str(value)+"\n")
            ms2_mz_int_pair_list.append(final_dict)
        targetDF = pd.DataFrame({"scan":scan_list,"precursorMZ":precursor_mz_list,"charge":charge_list,"Peptide_ID":pep_seq_list,"L_ID":pep_id_list,"ms2_mz_int_array":ms2_mz_int_pair_list,"peptide_seq_mod_info":peptide_seq_mod_list,"Protein":prot_list,"xcorr_best":xcorr_list,"RT":RT_list,"spectrums":spectrum_list})
        write_log ("  Done ...\n")

        return targetDF

#to check dot product of each psms against library for QC purpose

def normalizedDotProduct(p, q):
    num = np.dot(np.sqrt(p),np.sqrt(q))
    den1 = np.sum(p*p)
    den2 = np.sum(q*q)
    if den1 * den2 == 0:
        normDotProduct = 0
    else:
        normDotProduct = num/np.sqrt(den1*den2)
    
    #sometimes I see some values greater than 1 so we reduce them to 1
    if normDotProduct > 1:
        normDotProduct = 1
    return normDotProduct


# def normalizedDotProduct(e_intens,r_intens):
    
#     if np.sqrt( sum(e_intens*e_intens)*sum(r_intens*r_intens) )==0.0:
#         e_sim=0.0
#     else:
#         e_sim = sum(e_intens*r_intens)/np.sqrt( sum(e_intens*e_intens)*sum(r_intens*r_intens) )
    
#     return e_sim


#df is the dataframe following the consolidation of the psms (10 total)
#psmsDict is the dictionary that has the spectrum (exp.scan.charge) as the key and list of mzarray and intentity array as value

def computeDotProduct(df,psmsDict, Dscore_cutoff): 
    #dictionary initializing to update the dotproduct
    dp_dict = {}
    #noDP_cutoffPass list updates if no psms associated to L_ID have Dscore >= Cutoff
    #We will update such psms (capturing only one that has highest Jscore) and add later during extraction of data during reconsolidation
    noDP_cutoffPass = []
    dotProduct = []
    mz_cols = list(df.columns) #used for indexing the columns
    #just because numpy array is quicker than iterrows or itertuples
    np_arr = df.to_numpy() #dataframe converted to numpy array
    for row in np_arr:
        #define an empty list that updates Dscores . If list size is 0 that means no Dscore is more than 0.8
        selectDscoreList = []
        DscoreIndex = []
        #the consolidation has dictionary of ms2 mz and intensity so the keys are mz array and values are intensity arrays
        consolidatedMZ = list(row[mz_cols.index("ms2_mz_int_array")].keys()) #mzarray = key
        consolidatedIntensity = np.array(row[mz_cols.index("ms2_mz_int_array")].values()) #intensity array = value
        spectrumList = row[mz_cols.index("spectrums")] #these spectrums were updated just for computing dot product
        xcorrList = row[mz_cols.index("XCorr")] # this is the list of Xcorr 
#         print (consolidatedIntensity)
        for i,eachSpectrum in enumerate(spectrumList):
            specMZ = list(psmsDict[eachSpectrum][0]) #extracting mz for each spectrum in spectrum list
            specInt = list(psmsDict[eachSpectrum][1]) #extracting intensity for each spectrum in spectrum list
            
            #defining empty dictionary
            specMZInt = {"mz":[],"intensity":[]}
            for index,mz in enumerate(consolidatedMZ):
                if mz not in specMZ:
                    
                    specMZInt["mz"].append(mz)
                    #dont put 0, 0.0 is float and recognized by numpy properly
                    specMZInt["intensity"].append(0.0) #giving 0.0 intensity for missing mz value
                else:
                    get_index = specMZ.index(mz)
                    specMZInt["mz"].append(mz)
                    #using index - get_index to get the corresponding intensity of mz
                    specMZInt["intensity"].append(specInt[get_index])
#                 print (consolidatedIntensity)
            #normalized dot product

            spec_query = np.array(tr_featSpec["intensity"])
            spec_reference = np.array(lib_mz_int_dict["intensity"])

            dp = normalizedDotProduct(consolidatedIntensity,np.array(specMZInt["intensity"]))               
            dp_dict[eachSpectrum]= dp
            if dp > Dscore_cutoff:
                selectDscoreList.append(dp)
                DscoreIndex.append(i)

        if len(selectDscoreList) == 0:
            jscoreIndex = xcorrList.index(np.max(xcorrList))
            #use this index to extract psms spectrum information from spectrum list
            considerPSMS = spectrumList[jscoreIndex]
            noDP_cutoffPass.append(considerPSMS)
        
    return dp_dict, noDP_cutoffPass
    

def dotProductFrequencyLibrary(matched_df, xaxis, figname): 
    minv = 0
    maxv = 1
    bins = np.linspace(minv,maxv)

    plt.rcParams.update({'font.size': 10})
    fig,ax = plt.subplots(figsize=(4,2))
    plt.yticks(color="black")
    # size, scale = 1000, 10

    commutes2 = matched_df[xaxis]
    commutes2.plot.hist(grid=False, bins=bins, rwidth=0.9,
                   color='#F4F6F7',edgecolor='black', linewidth=1.0)
    
    plt.title('')
    plt.xlabel("Dscore")
    plt.ylabel('Frequency')
    
#     labels = ["0","1","2","3","4","5",">5"]
#     ax.set_xticklabels(labels)
# #     plt.legend([label1, label2],loc="best")
    figurename = figname+".pdf"
    figurename1 = figname+".png"
    fig.savefig(figurename, bbox_inches="tight", dpi=600 )
    fig.savefig(figurename1, bbox_inches="tight", dpi=600 )



################## Pre evaluation of PSMS using Dscore. ############

#this function is required for initial screening of PSMS. After consolidation, we evaluate the dot product of each psms
#we apply the dot product cutoff to generate the list of spectrum which is later pushed through cosensusPSMTable
def pre_cosensusPSMTable(idtxtFile, psmsDict, precDict, jump_mod_dict, sta_AA, exp_list):
    write_log ("  Constructing a consensus table for unique psms\n")

    idtxtdf_1 = pd.read_csv(idtxtFile, delimiter=";", skiprows=return_skiprows(idtxtFile, ";", "Peptide"))
    idtxtdf_1["spectrum"] = idtxtdf_1.apply(createOutfile, df=idtxtdf_1, axis=1)
    idtxtdf_1[["exp","scan","charge"]] = idtxtdf_1["spectrum"].str.split(".",expand=True)
    idtxtdf =  idtxtdf_1.loc[idtxtdf_1.exp.isin(exp_list)]
    idtxtdf.drop_duplicates(subset="spectrum", inplace=True, keep="first")
    
    write_log ("Total PSMS in all fractions = ",idtxtdf.shape[0])

    idtxtdf["Peptides"] = idtxtdf["Peptide"]
    
    #For modification or PTMs information on the consensus library
    idtxtdf[["plain_peptide","modifications"]] = idtxtdf.apply(computeModifications, jump_mod_dict=jump_mod_dict,sta_AA=sta_AA,axis=1)
    idtxtdf["massPosDict"] = idtxtdf.modifications.apply(lambda x: spectrumToDict(x))
    
    idtxtdf["PeptideSeqWithRealDelMass"] = idtxtdf.apply(lambda x: addModValInPepSeq(x.plain_peptide,x.massPosDict), axis=1)
    
    idtxtdf["mz_int_pairs"] = idtxtdf.spectrum.map(psmsDict)
    idtxtdf["precursorMZ"] = idtxtdf.spectrum.map(precDict)

    #applies QC program to get more confident library if ms2 matches < 3 remove all and if ms2 matches < 4 and Jscore < 20 throw
    write_log ("\nAll the PSMs with JScore < 30 are removed from our further analysis to increase the robustness of the library\n")
    
    idtxtdf1 = idtxtdf.loc[idtxtdf.XCorr.astype("float") > 30]
    idtxtdf2 = idtxtdf1.sort_values(by=["XCorr"], ascending = False) #sort by xcorr to select top 10 psms later
    

    idtxtdf2["L_ID"] = idtxtdf2.PeptideSeqWithRealDelMass+";"+idtxtdf2.charge+";"+idtxtdf2.precursorMZ
    
    
    #count the number of psms
    idtxtdf2['count'] = idtxtdf2.groupby('L_ID')['L_ID'].transform('count')
    #make a dictionary of L_ID and psms#
    dict_id = dict(zip(idtxtdf2['L_ID'] , idtxtdf2['count']))
    
    requiredCols = ['mz_int_pairs', 'L_ID','XCorr','spectrum']
    idtxtConsensus = idtxtdf2[requiredCols]
    
    idtxtConsensus2=idtxtConsensus.groupby('L_ID', as_index=False).agg(list)

    write_log (  "Done ...\n")

    return idtxtConsensus2

#this is preconsolidation function. 

def pre_psmConsolidate(df, topPsmCnt=10): #topPsmCnt is a new parameter with default =  10
    write_log ("  psms consolidation begin for top {} PSMs".format(topPsmCnt))

    mz_cols = list(df.columns)
    xcorr_list = []
    spectrum_list = []
    L_ID_list = []
    mzIntPairs_list = []
    
    for row in df.itertuples():
        xcorr = row[mz_cols.index("XCorr")+1]
        spectrum = row[mz_cols.index("spectrum")+1]
        L_ID = row[mz_cols.index("L_ID")+1]
        mz_int_pairs = row[mz_cols.index("mz_int_pairs")+1]
        

        xcorr_list.append(xcorr)
        spectrum_list.append(spectrum)
        L_ID_list.append(L_ID)
        
        mzIntPairs_list.append(mz_int_pairs)
        
    xcorr_list_top10 = []
    spectrum_list_top10 = []
    L_ID_list_top10 = L_ID_list #since this is only one ID and it is grouped by this value
    mzIntPairs_list_top10 = []
    
    
    for x in range(0,len(xcorr_list)):
        xcorr_list_new = xcorr_list[x][0:int(topPsmCnt)] #top 10 psms .. now changed to a parameter with topPSMcount
        xcorr_list_top10.append(xcorr_list_new)
        
        spectrum_list_new = spectrum_list[x][0:int(topPsmCnt)] #top 10 psms.. now changed to a parameter with topPSMcount
        spectrum_list_top10.append(spectrum_list_new)
        
        L_ID_list_new = L_ID_list[x][0:int(topPsmCnt)]
        L_ID_list_top10.append(L_ID_list_new)
        
        mzIntPairs_list_new = mzIntPairs_list[x][0:int(topPsmCnt)]
        mzIntPairs_list_top10.append(mzIntPairs_list_new)
        
    newDF = pd.DataFrame(list(zip(mzIntPairs_list_top10,
                                  L_ID_list_top10,
                                  xcorr_list_top10,
                                  spectrum_list_top10)),
                                  
                                  columns = ['mz_int_pairs', 'L_ID','XCorr','spectrum'])
    
    
    write_log ("  Top {} psms from all fractions are now consolidated for DScore evaluation\n".format(topPsmCnt))

    return newDF



#this functions combine the product ions to intensities together for top 10 psms
#thus made library is used for dot product computation
#This consolidates to generate the library and later cross evaluate the psms against the library to generate Dscore
#If the Dscore is less than Dscore cutoff we remove those psms 
#Following this we redo the consensus step with ID.txt removed from those spectrum
def pre_combine_mz_int(df):
    mz_cols = list(df.columns)
    

    #idea is to convert two list to one dictionary and keep updating dictionary as it sees another dictionary
    # keys = ['a', 'b', 'c']
    # values = [1, 2, 3]
    # dictionary = dict(zip(keys, values))
    #https://stackoverflow.com/questions/209840/how-do-i-convert-two-lists-into-a-dictionary
    #Update one dictionary from another https://stackoverflow.com/questions/10461531/merge-and-sum-of-two-dictionaries
    #x = {'both1': 1, 'both2': 2, 'only_x': 100}
    #y = {'both1': 10, 'both2': 20, 'only_y': 200}
    #print {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
    
    #making list for the decoy library generation

    pep_id_list = [] #list of peptide ID 
    ms2_mz_int_pair_list = [] #list of list of ms2 mz and int dictionary
#         ms2_int_list = [] #list of list of intensity
    xcorr_list = []
    spectrum_list = []

    write_log ("Consolidating the intensity of product ions for all PSMS of a peptide together into a dictionary")
    for row in df.itertuples():
        mzIntDict = {}
        xcorr = row[mz_cols.index("XCorr")+1]
        L_ID = row[mz_cols.index("L_ID")+1]
        mz_int_pairs = row[mz_cols.index("mz_int_pairs")+1]
        spectrums = row[mz_cols.index("spectrum")+1]
        
        mz_list_new = []
        int_list_new = []
        
        
        for x in range(0,len(mz_int_pairs)):
            each_pair = mz_int_pairs[x]
    #         print (each_pair[0]) #mz value
    #         print (each_pair[1]) #intensity value
            for index,val in enumerate(each_pair[0]):
                mz_list_new.append(float(val))
                int_list_new.append(int(each_pair[1][index]))

            dictionary = dict(zip(mz_list_new, int_list_new)) #make dictionary using mz as keys and intensity as values
            mzIntDict = {k: mzIntDict.get(k, 0) + dictionary.get(k, 0) for k in set(mzIntDict) | set(dictionary)}
        
#             print ("JScore = ",";".join(xcorr_str), "total ms2 spectra = ",len(mzIntDict.keys()))
        


        pep_id_list.append(L_ID)

        xcorr_list.append(xcorr)
        spectrum_list.append(spectrums)
        
        
        final_dict = collections.OrderedDict(sorted(mzIntDict.items()))

        ms2_mz_int_pair_list.append(final_dict)

    targetDF = pd.DataFrame({"L_ID":pep_id_list,"ms2_mz_int_array":ms2_mz_int_pair_list,"XCorr":xcorr_list,"spectrums":spectrum_list})
    write_log ("  Done ...\n")

    return targetDF
