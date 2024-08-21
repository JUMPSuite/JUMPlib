import pandas as pd 
import re
import numpy as np
import math
import pyteomics
from pyteomics import mass
from normalization_PSMSHandler import *
from logFunctions import *
pd.options.mode.chained_assignment = None  # default='warn'

#target decoy
def typePeptide(row):
    peptide = row.Peptide_ID
    if "Decoy" in peptide:
        pepType = "Decoy"
    else:
        pepType = "Target"
        
    return pepType

def plainPeptide(row):
    peptide = row.Peptide_ID
    if "Decoy_" in peptide:
        peptide = row.Peptide_ID.split("_")[1]
    pepNoFlankList = list(peptide) #flanking peptide is converted to the list, so the symbols are also member of list along with amino acids

    plain_peptide_list =[] #iniiates plain peptide formation
    for aa in pepNoFlankList:
        if aa in pyteomics.parser.std_amino_acids: #this looks at the standard amino acids in pyteomics and if the value in list is not amino acid for example * or other symbols, then it will discard those and just adds amino acdis
            plain_peptide_list.append(aa) 
    plain_peptide = "".join(plain_peptide_list) #creates a new string or amino acids
    return pd.Series([plain_peptide, len(plain_peptide)])



def TMTorNot(sta_AA):
    if sta_AA["K"] == sta_AA["n"]:
        result = "TMT"
    else:
        result = "NonTMT"
    return result


#this function processes search results and extracts RT for the peptides infering MS2 based scans





def postsearchProcessing(expDF, final_result, outputFolder,tmt,logFile, exp_ms2, L_ID_allProtDict,L_ID_prevAADict,L_ID_nextAADict):
    
    expDF["key"] = expDF.scan.astype("str")+"."+expDF.charge.astype("str")+"."+expDF.prec_MZ.astype("str")
    expDF["simMS2"] = expDF["key"].map(final_result)
    
    colKeep = ['key','scan', 'charge', '[M+H]+', 'prec_MZ','simMS2']
    expDF_simMS2 = expDF.copy(deep=False)[colKeep]
    expDF_simMS2_dropNA_simMS2 = expDF_simMS2.dropna(subset=["simMS2"])
    expDF_simMS2_dropNA = expDF_simMS2_dropNA_simMS2.copy(deep=False)
    #print (expDF_simMS2_dropNA)
    expDF_simMS2_dropNA["simMS2_Ranked"] = expDF_simMS2_dropNA.apply(rankMatchedPSMS, axis=1)
    expDF_simMS2_dropNA_split = tidy_split(expDF_simMS2_dropNA, "simMS2_Ranked", sep=',', keep=False)

    #we now split simMS2_Ranked

    expDF_simMS2_dropNA_split[["L_ID","Library_Match_score (DP)","RT","Peptide_ID","Charge_check","Prec_MZ_theoretical","Rank (PSMS)","deltacn","num_matched_ions","tot_num_ions"]]=expDF_simMS2_dropNA_split.simMS2_Ranked.str.split(";",expand=True) 

    #convert string scores to float scores
    expDF_simMS2_dropNA_split["JDscore"] = expDF_simMS2_dropNA_split["Library_Match_score (DP)"].astype("float")

    printCols = ['scan', 'charge', '[M+H]+', 'prec_MZ', 'L_ID', 'RT', 'Peptide_ID','Rank (PSMS)','Prec_MZ_theoretical', 'JDscore','deltacn','num_matched_ions','tot_num_ions']
    printDF = expDF_simMS2_dropNA_split[printCols]
    printDF[["plain_peptide","pepLength"]] = printDF.apply(plainPeptide, axis=1)
    printDF2 = printDF.sort_values(by=["JDscore"], ascending=False)
    
    #remove .ms2 from filename
    printDF2["exp"] = exp_ms2.split(".")[0]
    # printDF2["Peptide"] = printDF2.apply(makeJUMP_likePeptide,sta_AA=sta_AA,jump_mod_dict=jump_mod_dict,axis=1)
    
    printDF2["spectrum"] = printDF2["exp"]+"."+printDF2["scan"].astype("str")+".1."+printDF2["charge"].astype("str")
    #map protein to the unique peptide ID using the peptide Protein dictionary
    printDF2["Protein"] = printDF2.Peptide_ID.map(L_ID_allProtDict)
    #calculate experimental neutral mass
    printDF2.Prec_MZ_theoretical = printDF2.Prec_MZ_theoretical.astype("float")
    # print (printDF2.Prec_MZ_theoretical)
    printDF2["calcMH"] = printDF2.apply(lambda x: calcNeutralMass(x.Prec_MZ_theoretical,x.charge), axis=1)

    printDF2["Type"] = printDF2.apply(typePeptide, axis=1)
    printDF2["ppm"] = printDF2.apply(lambda x: ppmCalc(x.Prec_MZ_theoretical,x.prec_MZ), axis=1)
    printDF2["abs_dPrecMZ"] = printDF2["ppm"].abs()
    printDF2[["plain_peptide","pepLength"]] = printDF2.apply(plainPeptide, axis=1)
    printDF2["pep_prev_aa"] = printDF2.Peptide_ID.map(L_ID_prevAADict)
    printDF2["pep_next_aa"] = printDF2.Peptide_ID.map(L_ID_nextAADict)

    ###### RENAME COLUMNS FOR OUTPUT FILES ###############
    renameColsDict = {"Peptide_ID":"Peptide","spectrum":"Outfile","[M+H]+":"measuredMH"}

    printDF2.rename(columns=renameColsDict, inplace=True)
    displayCols = ["scan","Peptide","Protein","Outfile","measuredMH","calcMH","ppm","JDscore","abs_dPrecMZ","plain_peptide","pepLength", "L_ID", "RT","Rank (PSMS)", "Type","deltacn","num_matched_ions","tot_num_ions","pep_prev_aa","pep_next_aa"]



    printDF2[displayCols].to_csv(outputFolder+"/"+outputFolder+".allRanks.csv",index=None)
    # printDF2.to_excel(outputFolder+"/Library_Search_All_Ranks.xlsx",index=None)
    #select rank1 for histogram
    printDF2Rank1 = printDF2.copy(deep=False).loc[printDF2["Rank (PSMS)"] == "Rank1"]
    

    
    write_log (logFile,"\nNumber of PSMs (Rank 1) before consolidation = ", printDF2Rank1.shape[0])

    # #make one scan one PSM for TMT data else you can keep multiple PSMs for one scan
    # if tmt == "1":
    #     printDF2Rank1 = onePsmsOneMS2(printDF2Rank1)
    #     write_log (logFile,"\nNumber of PSMs (Rank 1) after consolidation = ", printDF2Rank1.shape[0])
    #     printDF2Rank1[displayCols].to_excel(outputFolder+"/"+outputFolder+".1.xlsx",index=None)
    #     # printDF2Rank1.to_excel(outputFolder+"/Library_Search_Rank1.xlsx",index=None)
    
    printDF2Rank1[displayCols].to_csv(outputFolder+"/"+outputFolder+".1.csv",index=None)
    
    write_log (logFile,"Library search results saved as ", outputFolder+"/"+outputFolder+".1.csv\n")
    #we need to work this out for non-labeled data ... some columns will be missing
    return printDF2Rank1, printDF2


#one psms is assigned to one ms2 scan
def onePsmsOneMS2(df): 
    reqCols = ['scan','charge','JDscore','Peptide','pepLength']

    df2 = df.copy(deep=False)[reqCols]

    df3 = df2.groupby('scan').agg(lambda x: x.tolist()).reset_index()
    df['psmConsolidateKey'] = df[["scan","JDscore","charge","Peptide"]].apply(
        lambda x: '_'.join(x.dropna().astype(str)),
        axis=1
    )

    psmKey = psmConsolidate(df3)
    df_consolidate = df.loc[df.psmConsolidateKey.isin(psmKey)]
    return df_consolidate

def duplicates(lst, use = "max"):
    if use == "max":
        return [i for i, x in enumerate(lst) if x == np.max(lst)]
    else:
        return [i for i, x in enumerate(lst) if x == np.min(lst)]

#first check maximum score
#second check minimum charge
#third look at minimum peptide length
def psmConsolidate(df):
    keyList = []
    cols = list(df.columns)
    np_arr = df.to_numpy()
    
    for pep_row in np_arr:
        
        scan = pep_row[cols.index("scan")]
        chargeList = pep_row[cols.index("charge")]
        scoreList = pep_row[cols.index("JDscore")]
        aaLenList = pep_row[cols.index("pepLength")]
        pepID = pep_row[cols.index("Peptide")]

        indices = duplicates(scoreList)
        if len(indices) == 1:
            reqdCharge = chargeList[indices[0]]
            reqdScore = scoreList[indices[0]]
            reqdpepID = pepID[indices[0]]
        if len(indices) > 1:
            chargeIndices = duplicates(chargeList,"min")
            if len(chargeIndices) == 1:
                reqdCharge = chargeList[chargeIndices[0]]
                reqdScore = scoreList[chargeIndices[0]]
                reqdpepID = pepID[chargeIndices[0]]
            else:
                aaLenIndices = duplicates(aaLenList,"min")
                reqdCharge = chargeList[aaLenIndices[0]]
                reqdScore = scoreList[aaLenIndices[0]]
                reqdpepID = pepID[aaLenIndices[0]]
                if len(aaLenIndices) > 1:
                    print ("\nPrinting Multiple PSMs that have same scores, same charge state and same peptide length. So, the program automatically selects the first PSM\n")
                    print ("Scan = ",str(scan))
                    print ("Score = ",str(reqdScore))
                    print ("Charge = ",str(reqdCharge))
                    print ("Peptide = ", reqdpepID)
                    # print (str(scan)+"_"+str(reqdScore)+"_"+str(reqdCharge)+"_"+reqdpepID)
                    print ("\n")
        key = str(scan)+"_"+str(reqdScore)+"_"+str(reqdCharge)+"_"+reqdpepID
        keyList.append(key)
    return keyList


