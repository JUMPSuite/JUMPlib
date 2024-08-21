import pandas as pd
import os, sys, glob
from consensusTarget import *
from consensusDecoy import *
from idtxtMs2ModsFunctions import *
from RTfunctions import *

from lowess import *
from logFunctions import *
import re
import collections #for ordering the dictionary
import pickle
import fileinput
from os.path import dirname

import configparser
import time
import math

start_time = time.time()

config = configparser.ConfigParser()

# # Read the parameter file using configparser

params_file = sys.argv[1]
# mzFILE = sys.argv[2]

config.read(params_file)

#unimod modification information
unimod_modPklFile = config["specLib"]["unimod_modPklFile"]
 
#example pepxml file to parse the modification information
pepxml = config["specLib"]["pepxml"]
#results directory where mass correction results are present
resultsDirectory = config["specLib"]["masscal_result"]
#jump -f results ID.txt
idtxt = config["specLib"]["jump_f_dyn"]
#all_peptide table
id_all_pep = config["specLib"]["id_all_pep"]
#mzxml path that contains folder that have mzxml files
mzxml_path = config["specLib"]["mzxml_path"]
#ordered fractions list
orderedFraction = config["specLib"]["orderedFraction"]
#Output folder name that contains spectral library files
specLibFolder = config["specLib"]["output_specLibFolder"]

#clustering
eps = float(config["specLib"]["eps"])

#decoy generation distance
distanceDecoy = float(config["specLib"]["distanceDecoy"])

#infered RT lookup table
rtFile_extract = config["specLib"]["rtFile_extract"]

#infered RT file (pkl) path
rtFilePath_extract = config["specLib"]["rtFilePath_extract"]


#Top PSMs to be considered 
topPsmCnt = config["specLib"]["topPsmCnt"]

#libary notes for example source of data, batch information etc
libraryNotes = config["specLib"]["libraryNotes"]

#QC parameter
Dscore_cutoff = float(config["specLib"]["Dscore_cutoff"])

#this is for naming the library. This depends on type of data
#1= tmt18_default, 2 = tmt11_default, 3= tmt18_pho, 4 = tmt11_pho, 5= tmt18_ub, 6 = tmt11_ub, 100 = labelfree_default, 1000 = silaclys_default

libtype = float(config["specLib"]["libtype"])



#this dictionaryneeds to be updated with other datatype
libtypeDict = {1:"tmt18_default", 2:"tmt11_default", 3:"tmt18_pho", 4:"tmt11_pho", 5:"tmt18_ub", 6:"tmt11_ub", 100:"labelfree_default", 1000:"silaclys_default"}
libtypename = libtypeDict[int(libtype)]

decoy_gen_method = config["specLib"]["decoy_gen_method"]

if (decoy_gen_method != "1") and (decoy_gen_method != "0"):
    print ("Please provide valid decoy generation strategy. Select 0 for precursor swap technique and 1 for mass shift")
    sys.exit(1)

path = os.getcwd()
os.chdir(path)

# make speclib folder
mkdir(specLibFolder)

#make intermediate folder to save intermediate library files to 
# make speclib folder
mkdir(specLibFolder+"/intermediate")


#get the unimod ptm dictionary unimod_mod_infor using the pickle file from unimod
with open(unimod_modPklFile, 'rb') as unimod_handle:
    unimod_mod_infor = pickle.load(unimod_handle)

write_log ("\n\n******** JUMP -lib -d program *********\n")
write_log ("  NOTE: You need to run JUMP -mc program to get\n the mass corrected ions that are matched with theoretical ions that makes the library !!!\n")


write_log ("\n  Obtaining modification function using the template pepXML file\n. So, it is important you provide the pepXML file representing the current batch\n")
jump_modAA_dict, jump_mod_dict, sta_AA = getDynStatModsInfoPepXml(pepxml)

write_log ("  Retrieving all theoretical ions matched after correction of all ions\n")

#This folder has all ms2 fractions for example 120 for reference
try:
    ms2ListAll = glob.glob(resultsDirectory+"/ms2/*.theo.ms2pep") #all theoretical ions for consensus

except:
    write_log ("  Please make sure you have results from JUMP -mc and correct path is provided in the parameters!!!")
    sys.exit(1)

#idtxt = "sum_jumpf/ID.txt"  #this we need for advanced annotation of L in ms2 file after consensus
#mzXMLs = glob.glob(mzxml_path+"/*/*.mzXML")

# ++++ raw_data_type ++++
raw_data_type = "mzXML"
if len( glob.glob(mzxml_path+"/*.mzML") )>0:
    raw_data_type = "mzML"

#check if mzxml file are in mzXML_possible
mzXML_possible = getOrderedMzxmlList(mzxml_path, orderedFraction,raw_data_type)

#this checks if the mzXML file is in the mzxml_path or inside the fraction folder in mzxml_path
mzXMLs = []
for mzfile in mzXML_possible:
    if os.path.exists(mzfile):
        mzXMLs.append(mzfile)
    else:
        fraction = os.path.split(mzfile)[-1].split(".{}".format(raw_data_type))[0]
        mzfile_2 = "{}/{}/{}.{}".format(dirname(mzfile),fraction,fraction,raw_data_type)
        mzXMLs.append(mzfile_2)

#keep only ms2 based on ordered fraction. In case someone wants to just use subset of fraction, we need to control program to generate subset ms2

ms2List = []
exp_list = []

for mzXML in mzXMLs:
    tail = os.path.split(mzXML)[-1].split(".{}".format(raw_data_type))[0]
    exp_list.append(tail)
    pickedMS2 = glob.glob(resultsDirectory+"/ms2/"+tail+".theo.ms2pep")[0]
    ms2List.append(pickedMS2)
    


write_log ("  The {} files are ordered for RT inference program.\nA total of {} files are used for libary generation\n".format(raw_data_type,len(mzXMLs)))

#Get RT information
#this is now changed based on Ji-Hoon's new program
#since the RT finding step takes time, we just save the entire RT table as csv file and just import that for next time

if rtFile_extract == "0" :
    ext_data_dict,res = inferRT(idtxt, mzXMLs, eps)

    for key in ext_data_dict.keys():
        ext_data_dict[key].to_pickle("{}/intermediate/{}.pkl".format(specLibFolder,key))

    res.to_csv(specLibFolder+"/intermediate/RT_all_fractions_after_inference.csv", index=None)
    res.to_pickle(specLibFolder+"/intermediate/RT_all_fractions_after_inference.pkl")
    print ("Time taken for RT inference --- %s seconds ---" % (time.time() - start_time))
    
else:
    print ("  RT extraction file is provided so RT extraction process is skipped and will proceed to RT alignment")
    res = pd.read_pickle(rtFilePath_extract)
    run_n_psm = [x+"_nPSMs" for x in exp_list]

    res = res[["key"]+exp_list+run_n_psm]

#suresh alignment
#df_lowess,xpred, ypred, mod = align_lowess(res, ref_run = exp_list[0])
#ji-hoon's alignment
df_lowess,deltaRT_recorder = alignRT(res, exp_list, eps)
df_lowess.to_csv(specLibFolder+"/intermediate/RT_all_fractions_after_alignment.csv", index=None)
df_lowess.to_pickle(specLibFolder+"/intermediate/RT_all_fractions_after_alignment.pkl")
#the rt estimated dataframe us generated from the csv file

#compute standard deviation of delta RT for each peptide

delRT_df = deltaRT_recorder[0]

for x in range(1, len(deltaRT_recorder)):
    delRT_df = delRT_df.merge(deltaRT_recorder[x], how="outer", on="key")
all_cols = delRT_df.columns.drop("key")
delRT_df["delRT_std"] = np.nanstd(delRT_df[all_cols], axis=1)
delRT_df.to_csv(specLibFolder+"/intermediate/delRT_each_peptide.csv", index=None)  

#save each file from the ext_data_dict as pickle file for future use
# print_cols = ['ms2_scan', 'prec_mz', 'prec_intensity', 'ms2_rt']




#This step takes a lot of time depending on the number of samples




#make RT dictionary for average RT
mzRT_dict = dict(zip(df_lowess.key, df_lowess[exp_list[0]]))

# save the dataframe as the pickle file so that for future alignment we can just tweak this file
# res.to_pickle(specLibFolder+"/RT_all_fractions.pkl")

# Store data (serialize)
#with open(specLibFolder+"/RT_all_fractions.pickle", "wb") as handle:
#    pickle.dump(mzRT_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

psmsDict,precDict = ms2fileToDict(ms2List)

write_log ("The program has initiated the consensus and consolidation step.")
write_log ("  This step creates a library and computes Dscore of each PSMS that has Jscore > 30")
write_log ("  The Dscore and spectrum are saved as the pickle file")
write_log ("  The cutoff of Dscore is applied. The spectrum that pass the cutoff are only used to perform the reconsensus")


write_log ("  All the unique psms are consolidated together as a table. An unique psms is made up of scan+charge+precMZ+fraction+modifications")
pre_idtxtConsensus2 = pre_cosensusPSMTable(idtxt, psmsDict, precDict, jump_mod_dict, sta_AA,exp_list)

write_log ("  Selection of top {} psms ongoing".format(topPsmCnt))
pre_idtxtConsensus3 = pre_psmConsolidate(pre_idtxtConsensus2, topPsmCnt)

write_log ("  Final intensity consolidation process ongoing")
pre_target_DF = pre_combine_mz_int(pre_idtxtConsensus3)

write_log ("  The pre-consolidation step is done.\n")
######### Computes Dscore ########
#QC of the library generation using dot product calculation of each psms against the library

write_log ("Initiating the QC. Calculating dot product of each psms against the library")
#Dscore cutoff application .. this is the parameter

write_log ("  Dscore calculation in underway. A Dscore cutoff = {} is applied".format(Dscore_cutoff))
spectrum_Dscore_dict, considerPSMS = computeDotProduct(pre_target_DF,psmsDict, Dscore_cutoff)

write_log ("  TYPE1: A total of {} PSMS did not have spectrum above Dscore above {}. The spectrum of maximum Jscore is selected for building libary. Please adjust Dscore cutoff if this list is very high".format(len(considerPSMS),Dscore_cutoff))
#This updates list that has Dscore > Dscore_cutoff


##### Generating plot for the Dscores for all spectrum #####

df=pd.DataFrame({"Spectrum":list(spectrum_Dscore_dict.keys()),"DScores":list(spectrum_Dscore_dict.values())})
write_log ("  The total psms for dot product QC is {}".format(df.shape[0]))

df2 = df.dropna()
write_log ("  The total psms for dot product QC is after removal of NA {}".format(df2.shape[0]))

write_log ("  Saving dot product values as the csv file")
df2.to_csv(specLibFolder+"/intermediate/Spectrum_Dotproduct.csv", index=None)

write_log ("  The lowest dot product value is {}".format(np.min(df2["DScores"])))

# #ploting of dot product computed for the library
dotProductFrequencyLibrary(df2,"DScores", specLibFolder+"/intermediate/QC_psms_dotProduct_histogram")



dscorePassSpec = []

#use Dscore cutoff to update the list of spectrum
for k,v in spectrum_Dscore_dict.items():
    if v > Dscore_cutoff:
        dscorePassSpec.append(k)

###################################

write_log ("  TYPE2:A total of {} spectrum have Dscore > {} and will be used for building library".format(len(dscorePassSpec),Dscore_cutoff))
write_log ("  TYPE1 and TYPE2 spectrum are appended together for building library\n")

#updating the library building list
library_build_spec = dscorePassSpec+considerPSMS
#saving the dictionary of spectrum and dotproduct as the pickle file for future reference to see why some psms were removed from analysis
with open(specLibFolder+'/intermediate/spectrumDscore.pickle', 'wb') as handle:
    pickle.dump(spectrum_Dscore_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


#find id_all_pep 
if id_all_pep == "0":
    id_all_pep = dirname(idtxt)+"/publications/id_all_pep.txt"


    
#Use the id_all_pep.txt file for generating the peptide to representative protein dictionary
pepToUniqProtDict = peptide_protein_map_library(id_all_pep, jump_mod_dict, sta_AA, specLibFolder)


write_log ("  \nA unique key composed of sample name, scan number and charge is matched with product ions and their intensity pairs")
write_log ("  Total keys in {} files = {}".format(str(len(ms2List)), len(psmsDict.keys())))
write_log ("  All precursor mz for each key is assigned as the different dictionary\n") 

#this is for idtxtConsensus protein peptide mapping
idtxtConsensus2 = cosensusPSMTable(idtxt, psmsDict, precDict, jump_modAA_dict, jump_mod_dict, sta_AA, unimod_mod_infor,mzRT_dict,specLibFolder, pepToUniqProtDict, spectrumList = library_build_spec)

idtxtConsensus3 = psmConsolidate(idtxtConsensus2, topPsmCnt)

# idtxtConsensus3.to_csv(specLibFolder+"/intermediate/idconsensus.csv")


# #make directory spectral library folder
# cmd1 = "mkdir "+specLibFolder
# try:
#   os.system(cmd1)
# except:
#   pass
write_log ("  Library notes {} are added. This is important to keep track of the database".format(libraryNotes))



targetDF = createMS2EachPSMS_L_ID(idtxtConsensus3, specLibFolder, libraryNotes, libtypename)
targetDF.to_pickle(specLibFolder+"/intermediate/jumplib_human_{}_target.pkl".format(libtypename))

#generate the decoy library using newly concatenated library

total_entries = targetDF.shape[0]

appendLibraryFinal_sorted = targetDF.sort_values(['precursorMZ','charge'], ascending=[True, True])
decoy_master_list = [] #decoy_master_dict[key1] = [decoy_scan, precMZ, charge, massNeutral, L_ID, L_peptide, RT,mz];decoy_master_dict[key2] = [decoy_scan_pair, precMZ_pair, charge, massNeutral_pair, L_ID_pair, L_peptide_pair, RT_pair,mz_pair]

if decoy_gen_method == "0": # 0 is precursor swap (default)

    total_blocks = math.ceil(total_entries/12000)
    

    # shuffled = appendLibraryFinal_sorted.sample(frac=1)
    result = np.array_split(appendLibraryFinal_sorted, total_blocks) 
    #decoySpecLibrary_Prec_Swap_New

    exclusion_list = []
    rescue_scans_list = []

    shape = appendLibraryFinal_sorted.shape[0]

    print ("..... Working for {} blocks of target library. Each block has ~12000 scans".format(total_blocks))

    for i,x in enumerate(result):
        print ("..... Working for {} index out of {} blocks of target library".format(i, total_blocks))
        # print ("The section of dataframe columns are {}, and shape of dataframe is {}".format(x.columns, x.shape[0]))
        exclusion_list, rescue_scans_list, decoy_dict = decoySpecLibrary_Prec_Swap_New(x, specLibFolder, distanceDecoy,libtypename, shape, exclusion_list, rescue_scans_list)
        decoy_master_list.append(decoy_dict)



    #check if all scans in rescue_scans_list is in exclusion_list
    rescue_scans_list = list(set(rescue_scans_list).difference(set(exclusion_list)))
    rescue_df_sorted = appendLibraryFinal_sorted[appendLibraryFinal_sorted["scan"].astype("int").isin(rescue_scans_list)]
    rescue_scans_list2 = []

    shape = appendLibraryFinal_sorted.shape[0]
    exclusion_list, rescue_scans_list2,decoy_dict = decoySpecLibrary_Prec_Swap_New(rescue_df_sorted, specLibFolder, distanceDecoy,libtypename, shape, exclusion_list, rescue_scans_list2)
    decoy_master_list.append(decoy_dict)


    decoy_dict = rescue_scan_decoy(appendLibraryFinal_sorted, rescue_scans_list2, distanceDecoy)
    decoy_master_list.append(decoy_dict)
    write_decoy_library(decoy_master_list, specLibFolder, distanceDecoy,libtypename)

else:
    #generate the decoy library using newly concatenated library
    decoySpecLibrary(appendLibraryFinal, specLibFolder, distanceDecoy,libtypename)

    # # 1 is precursor mass shift distanceDecoy is the mass shift that is used to generate decoy
    # rescue_scans_list2 = list(appendLibraryFinal_sorted.scan.astype("int"))
    # decoy_dict = rescue_scan_decoy(appendLibraryFinal_sorted, rescue_scans_list2, distanceDecoy)
    # decoy_master_list.append(decoy_dict)
    # write_decoy_library(decoy_master_list, specLibFolder, distanceDecoy,libtypename)

specLib = specLibFolder+"/intermediate/jumplib_human_{}_target.splib".format(libtypename)
# decoyLib = specLibFolder+"/intermediate/jumplib_human_{}_decoy.splib".format(libtypename)
decoyLib_list = glob.glob(specLibFolder+"/intermediate/jumplib_human_{}_decoy.splib".format(libtypename))
outfilename = specLibFolder+"/jumplib_human_{}.splib".format(libtypename)
filenames = [specLib]+decoyLib_list

#concatenate two spectral library
with open(outfilename, 'w') as fout, fileinput.input(filenames) as fin:
    for line in fin:
        fout.write(line)



cmd2 = "cp "+params_file+" "+specLibFolder
os.system(cmd2)

logFile = "jump_lib_db.log"
cmd3 = "mv "+logFile+" "+specLibFolder
os.system(cmd3)

write_log ("Generating library as the dataframe and store it as a pickle file. This is very important as parsing pickle file can save >1 minute per file during searching.")
libDF = ms2ToDf_spec(outfilename)
libDF.to_pickle(specLibFolder+"/jumplib_human_{}.pkl".format(libtypename))

write_log ("Time taken for generating spectral library database --- %s seconds ---" % (time.time() - start_time))

