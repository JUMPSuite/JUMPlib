
import pandas as pd
from normalization_PSMSHandler import *

from mainSearchFunctions import *
from postSearchProcessing import *
from RT_score import *

import re, numpy as np
import math
import os
import os.path
import matplotlib.pyplot as plt
import sys
import time
from logFunctions import *
pd.options.mode.chained_assignment = None  # default='warn'
from spectra_process import * # get_spec_df_from_pkl, get_spec_df_from_ms2, entropy_sim_per_peak, get_similarity, Convert_JUMPlib_csv2pepXML

import configparser
config = configparser.ConfigParser()


start_time = time.time()

# # Read the parameter file using configparser
# params_file = "/Users/spoudel1/Desktop/JUMP_specLib/Program/finalProgram/specLib_firstSearch.params"#sys.argv[1]
params_file = sys.argv[1]
exp_ms2 = sys.argv[2]

# outputFolder = exp_ms2.split("/")[-1].split(".ms2")[0]
outputFolder = exp_ms2.split("/")[-1].split(".")[0]

makedirectory(outputFolder)

config.read(params_file)

# exp_mzxml = config["specLib"]["exp_mzxml"] #raw file
# exp_ms2 = config["specLib"]["exp_ms2"] #raw file
specLibFolder = config["specLib"]["specLibFolder"] #library containing folder
ms1_tol = float(config["specLib"]["ms1_tol"]) #precursor ion tolerance

top_ions_control = config["specLib"]["top_ions_control"]
top_ions_min_max = config["specLib"]["top_ions"] #total number of top library ions that should be present in spectrum 
# top_ions_entry_rank = config["specLib"]["top_ions_entry_rank"] #total number of top library ions that should be present in spectrum 
ms2_tol = float(config["specLib"]["ms2_tol"]) #fragment ion tolerance
# rt_fdr = float(config["specLib"]["rt_fdr"]) #rt tolerance in SD
#window number determines the lenght of mz for normalization of intensity
window_number = int(config["specLib"]["window_number"])

binsize = config["specLib"]["binsize"] #binning width for selecting top ions in the ms2 spectrum 
top_ions_per_bin = int(config["specLib"]["top_ions_per_bin"]) #maximum fragment ions that are selected from raw spectrum for each bin
if str(binsize).upper() == "NO":
    binsize = 100
else:
    binsize = int(binsize)


start_end_scan_range = config["specLib"]["start_end_scan_range"] #start and end scans supplied by users


# outputFolder = fileroot+"/"+config["specLib"]["outputFolder"] #search results are stored in this folder

#score method
method = config["specLib"]["method"]
null_search = config["specLib"]["null_search"]
sim_mass = float(config["specLib"]["sim_mass"])
n_cores = int(config["specLib"]["n_cores"])
job_type = config["specLib"]["job_type"]

#check null search and if null_search = 1, sim_mass needs to be added else it is 0
if null_search == "0":
    sim_mass = 0.0

#tolerace type for dynamic or static tolerance

tolerance_type = config["specLib"]["tolerance_type"]

#Dynamice tolerance file if dynamic tolerance is selected

dyn_tol_file = config["specLib"]["dyn_tol_file"]


#tolerance for ms2 matching = fragment ion tolerance
#tol = float(config["specLib"]["tol"])
#example pepxml file to parse the modification information
tmt = config["specLib"]["tmt"]

#Let us first look at the all protein ppml file

libtype = float(config["specLib"]["libtype"])

min_top_ions = int(top_ions_min_max.split(",")[0])
top_ions = int(top_ions_min_max.split(",")[1])


# n_ions = int(top_ions_entry_rank.split(",")[0])
# topRanks = int(top_ions_entry_rank.split(",")[1])


#this dictionaryneeds to be updated with other datatype
libtypeDict = {1:"tmt18_default", 2:"tmt11_default", 3:"tmt18_pho", 4:"tmt11_pho", 5:"tmt18_ub", 6:"tmt11_ub", 100:"labelfree_default", 1000:"silaclys_default"}
libtypename = libtypeDict[int(libtype)]


# ++++ L_ID_allProtDict ++++
#take all protein ppml file first
all_prot_ppml = specLibFolder+"/intermediate/id_all_pep.ppml"
#make all protein ppml file dataframe
all_prot_ppmlDF = fileToDF(all_prot_ppml)
id_prot = all_prot_ppmlDF[["PeptideSeqWithRealDelMass","Protein Accession #"]]
# ++++ Combine_id_prot ++++
# Step 1: Create the decoy version of id_prot
Decoy_id_prot = id_prot.copy()
Decoy_id_prot["PeptideSeqWithRealDelMass"] = 'Decoy_' + Decoy_id_prot["PeptideSeqWithRealDelMass"]
Decoy_id_prot["Protein Accession #"] = 'Decoy_' + Decoy_id_prot["Protein Accession #"]
# Step 2: Combine the original and decoy versions into one dataframe
Combine_id_prot = pd.concat([id_prot, Decoy_id_prot], ignore_index=True)

#rename names for consistency and groupby peptide ID and get accession separated by ,
idProtDF = Combine_id_prot.groupby('PeptideSeqWithRealDelMass').agg( lambda x: ','.join(list(x))).reset_index()
#protein maped to peptide dataframe
# print (idProtDF.columns)
idProtDF.rename(columns = {"PeptideSeqWithRealDelMass":"Peptide","Protein Accession #":"Protein"}, inplace=True)
L_ID_allProtDict = dict(zip(idProtDF.Peptide, idProtDF.Protein))


# ++++ L_ID_prevAADict,L_ID_nextAADict ++++
#take uni protein ppml file first
uni_prot_ppml = specLibFolder+"/intermediate/id_uni_pep.ppml"
#make uni protein ppml file dataframe
uni_prot_ppmlDF = fileToDF(uni_prot_ppml)
id_pept = uni_prot_ppmlDF[["PeptideSeqWithRealDelMass","Peptides"]]
# get pep_prev_aa, pep_next_aa
id_pept["pep_prev_aa"] = id_pept.apply(lambda x: x["Peptides"].split(".")[0], axis=1)
id_pept["pep_next_aa"] = id_pept.apply(lambda x: x["Peptides"].split(".")[-1], axis=1)
id_pept.rename(columns = {"PeptideSeqWithRealDelMass":"Peptide"}, inplace=True)
id_pept_prev = id_pept[["Peptide","pep_prev_aa"]]
id_pept_next = id_pept[["Peptide","pep_next_aa"]]
L_ID_prevAADict = dict(zip(id_pept_prev.Peptide, id_pept_prev.pep_prev_aa))
L_ID_nextAADict = dict(zip(id_pept_next.Peptide, id_pept_next.pep_next_aa))


#set current directory as working directory
path = os.getcwd()
os.chdir(path)

# exp_ms2 = "/Users/spoudel1/Desktop/JUMP_specLib/Program/ms2/preprocess/FTLD_Batch2_F50.ms2"
specLib = specLibFolder+"/jumplib_human_{}.splib".format(libtypename)
# specLib = "/Users/spoudel1/Desktop/JUMP_specLib/specLib/OneFraction/SpectralLibraryTargetDecoy.spLib"


#expMZXML = exp_ms2.split("/")[-1].split(".")[0]


int_sd_dict = {}
if tolerance_type.upper() == "DYNAMIC":
    int_sd_dict = parseDynamicIntensityFile(dyn_tol_file,tol)


logFile = outputFolder+"/jump_lib_s.log"

#removing log file if previously present
try:
    rmFile(logFile)
except:
    print ("No jump_lib_s.log is present. Writing log file now")



write_log (logFile,"Parsing experimental ms2 file and storing as dataframe. Below is the summary")
write_log (logFile,"Name of input file = {}".format(exp_ms2))
write_log (logFile,"This will take time depending on size of ms2 file")

#simulated mass changes the precursor m/z by sim_mass dalton. This simulation is just done at the input file not in the library
# expDF_all = ms2ToDf_spec(exp_ms2, sim_mass)
expDF_all = get_spec_df_from_ms2(exp_ms2, 6, binsize, sim_mass)
if len(expDF_all["scan"])==0:
    sys.exit("")
write_log (logFile,"\nThe total number of precursor candidates (dtas) are {}".format(expDF_all.shape[0]))

write_log (logFile,"\nParsing Spectral Library database and storing as dataframe. Below is the summary")
write_log (logFile,"This will take time depending on size of ms2 file")



#check to see if the user has supplied start and end scans

start_end = start_end_scan_range.split(",")
start_scan = int(start_end[0])
end_scan = int(start_end[1])

#parse the expDF_all to get the start and end scans only

if end_scan != 0:
    # expDF = expDF_all.loc[(expDF_all.scan >= start_scan) & (expDF_all.scan <= end_scan)]
    expDF = expDF_all[expDF_all.scan.between(start_scan,end_scan)]
else:
    expDF = expDF_all


specLib_pkl = specLibFolder+"/jumplib_human_{}.pkl".format(libtypename)
#There should be a pickle file in the specLibFolder but just to make sure
if os.path.exists(specLibFolder+"/jumplib_human_{}.pkl".format(libtypename)):
    # libDF = pd.read_pickle(specLibFolder+"/jumplib_human_{}.pkl".format(libtypename))
    libDF = get_spec_df_from_pkl(specLib_pkl, 2, binsize)
    if len(libDF["scan"])==0:
        sys.exit("")
else:
    libDF = ms2ToDf_spec(specLib)
   #outfilename = specLibFolder+"/jumplib_human_{}.splib".format(libtypename) 
    libDF.to_pickle(specLibFolder+"/jumplib_human_{}.pkl".format(libtypename))



# maxv = np.max(expDF["prec_MZ"])+1
# minv = np.min(expDF["prec_MZ"])-1
# bins = np.linspace(minv,maxv, 1000)
# libDF["bin_prec_mz"] = pd.cut(libDF.prec_MZ, bins=bins)
# expDF["bin_prec_mz"] = pd.cut(expDF.prec_MZ, bins=bins)

write_log (logFile,"\nThe total TARGET entries in Library are {}".format(libDF.shape[0]/2))

# if method != "normalized_dot_product":
    # write_log (logFile,"Since the scoring method is not normalized dot product. We are normalizing the intensity. Performing normalization step\n")
    # normalizeIntensity(expDF,window_number)
    # normalizeIntensity(libDF,window_number)

# if tolerance_type.upper() == "DYNAMIC":
    # logTransformMS2Intensity(expDF)#this is for dynamic intensity tolerance

write_log (logFile,"\nLibrary searching is underway. Below are the list of major steps performed")
write_log (logFile,"Matching Precursor ions with ms1 tolerace (ppm) = {}".format(ms1_tol))
write_log (logFile,"Matching product ions")
write_log (logFile,"Searching for top matching fragment ions {}".format(min_top_ions))
write_log (logFile,"If found, cleaning the library entries for redundancy")
write_log (logFile,"Preprocessing input spectra for pattern matching")
write_log (logFile,"For each library product ions, look for ions with tolerance (ppm) = {}".format(ms2_tol))
write_log (logFile,"If no match found intensity = 0 to that product ion\n\n")



'''
#splitting to multiple frames
##Test this

###Parameters###
n_ions = 10
topRanks = 5

final_result = {}

df_split = np.array_split(expDF, 3)
cnt=1
for x in df_split:
    result = search(x,libDF,n_ions, ms2_tol,ms1_tol, int_sd_dict,topRanks,method="normalized_dot_product")
    print ("TOtal frames completed {}".format(cnt))
    cnt+=1
    final_result.update(result)

'''

#multipooling

#library searching results are kept as a dictionary final result
# final_result = librarySearchMain(expDF,libDF,top_ions,ms2_tol,ms1_tol,top_ions_per_bin,binsize,tolerance_type,int_sd_dict, method, sim_mass)
# final_result = librarySearchMain(expDF,libDF,top_ions,ms2_tol,ms1_tol,top_ions_per_bin,binsize,tolerance_type,int_sd_dict, method

#use multiprocessing here
#Define arguments that goes after the main dataframe that is going to be searched expDF
final_result = {}

if (job_type == "0") & (n_cores > 1):
    # args = (libDF,quick_dscore_check, n_ions, topRanks,ms2_tol,ms1_tol,top_ions_per_bin,binsize,tolerance_type,int_sd_dict, method)
    args = (libDF,top_ions_control, min_top_ions, top_ions,ms2_tol,ms1_tol,top_ions_per_bin,binsize,tolerance_type,int_sd_dict, method)

    #initialize empty dictionary to update after multiple processing
    
    #calling the parrallel function. result variable will be the list of result of 4 processes
    #input is the input DF for searching and remaining arguments args
    
    result = parallelize_dataframe(librarySearchMain, expDF, args,  n_cores=n_cores)
    for x in result:
        final_result.update(x.get()) #main work of the workers. The result is updated to the final result



    # split_n = math.ceil(16/n_cores) #this will help split the dataframe based on number of cores available
    # df_split = np.array_split(expDF, split_n)
    # for df_x in df_split:
    #     result = parallelize_dataframe(librarySearchMain, df_x, args,  n_cores=n_cores)
    #     for x in result:
    #         final_result.update(x.get()) #main work of the workers. The result is updated to the final result

else:
    # final_result = librarySearchMain(expDF,libDF,quick_dscore_check, n_ions, topRanks,ms2_tol,ms1_tol,top_ions_per_bin,binsize,tolerance_type,int_sd_dict, method)

    df_split = np.array_split(expDF, 5)
    cnt=1
    for x in df_split:
        result = librarySearchMain(x,libDF,top_ions_control, min_top_ions, top_ions,ms2_tol,ms1_tol,top_ions_per_bin,binsize,tolerance_type,int_sd_dict, method)
        print ("TOtal frames completed {}".format(cnt))
        cnt+=1
        final_result.update(result)


write_log (logFile,"Total candidates psms (all Ranks) before filtering = {}".format(len(final_result.keys())))


# consolidation of one psms to one scan for TMT data
printDF2Rank1, printDF2 = postsearchProcessing(expDF, final_result, outputFolder,tmt,logFile, exp_ms2, L_ID_allProtDict,L_ID_prevAADict,L_ID_nextAADict) #sta_AA helps us to see if the data is TMT or not {'K': 304.2071453, 'C': 57.02146, 'n': 304.2071453} #for tmt K and n values are equal

#### RT Infer and Score ####
searchFile = outputFolder+"/"+outputFolder+".1.csv"

#mzxml = exp_ms2.split(".ms2")[0]+".mzXML"
mzxml_1 = exp_ms2.split(".ms2")[0]+".mzXML"
mzxml_2 = exp_ms2.split(".ms2")[0]+".mzML"
mzxml = mzxml_1
if os.path.isfile( mzxml_2 ):
    mzxml = mzxml_2

# print (mzxml)
rt_score(searchFile, mzxml, outputFolder, logFile)


# Convert_JUMPlib_csv2pepXML
searchFile_all = outputFolder+"/"+outputFolder+".allRanks.csv"
Convert_JUMPlib_csv2pepXML(searchFile_all)


cmd2 = "cp "+params_file+" "+outputFolder
os.system(cmd2)


write_log (logFile,"The JUMPp-lib search is complete for {}".format(exp_ms2))
write_log (logFile,"Time taken for library search --- %s seconds ---" % (time.time() - start_time))
write_log (logFile,"\n\n")
