import numpy as np
# import pprint

import os
# import sys
import pickle
import pandas as pd
import re
# import time
import math
# from numba import jit

import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime



#### 0. def
# Change_ext
def Change_ext(original_fullfile,new_ext):
    # change the extension of fullfile
    [tpath,tname] = os.path.split(original_fullfile)
    [fname,ext] = os.path.splitext(tname)
    new_name = '%s%s' % (fname,new_ext)
    new_fullfile = os.path.join(tpath,new_name)
    return new_fullfile

# @jit(nopython=True)
# preprocess_topk_per_100mz
def preprocess_topk_per_100mz(exp_mz, exp_int, top_ions, binsize):# top_ions = 10 # take highest top 10 ions using intensity to rank
    # 1. cut off between (cutoffLowMass, cutoffHighMass)
    norm_1e2 = 1e2 # RT
    norm_1e4 = 1e4 # prec_MZ
    cutoffLowMass = 176 # Low Mass cutoff
    cutoffHighMass = 2000 # high mass limit
    exp_mz = np.array(exp_mz)
    exp_int = np.array(exp_int)
    ix = np.nonzero((exp_mz>=cutoffLowMass) & (exp_mz<=cutoffHighMass))[0] # cut off by range of cutoffLowMass and cutoffHighMass
    mz_list = exp_mz[ix].tolist()
    intensity_list = exp_int[ix].tolist()
    
    # 2. binning_mz_100
    top_mz_list = []
    top_intensity_list = []
    
    maxv = max(mz_list)+binsize
    minv = min(mz_list)-binsize
    bins = np.arange(minv, maxv, binsize) # fixed bin width 100 mz
    
    start_val = 0
    for x in range(1,len(bins)):
        sub_list_mz = [mzval for mzval in mz_list if bins[x-1] <= mzval < bins[x]] # subset mz list w
        if len(sub_list_mz) >=1: # some bins may not have any ions so we need to check the bin's element
            index_int = mz_list.index(sub_list_mz[-1]) # index for last value of sub list mz
            sub_list_int = intensity_list[start_val:index_int+1] # extract intensity until last value of sub list using intensity list (start and end index)
            start_val = index_int+1 # update start value for next round
            
            ind = np.argsort([-i for i in sub_list_int]) # sorting by descending order intensity
            r1 = np.argsort(ind)
            ix=np.nonzero(r1<top_ions)[0]
            
            # ind = np.argsort(sub_list_int) # sorting by descending order intensity
            # r1 = np.argsort(ind)
            # ix=np.nonzero(r1>=len(sub_list_int)-top_ions)[0]
            
            top_ion_lib=np.array(sub_list_mz)[ix].tolist()
            top_int_lib=np.array(sub_list_int)[ix].tolist()
            
            top_mz_list+=top_ion_lib  # append top 6 ions in the bin
            top_intensity_list+=top_int_lib # append top 6 ions intensity in the bin
    
    # 3. the intensity used are throughout normalized for the dot product calculation
    # simplifiedIonsDict = {"mz":top_mz_list,"intensity":top_intensity_list} # this dictionary stores mz and intensity after selecting top ions (example 10) after binning into 100 mz window
    mz=norm_1e4*np.array(top_mz_list)
    mz=[int(x)*1.0/norm_1e4 for x in mz]
    inten=np.array(top_intensity_list)
    
    # norm by the highest peak or the 2-fold of sum of peaks
    topinten_highest=max(inten)
    topinten_2sum=2*sum(inten)
    
    # norm by the highest peak
    inten=norm_1e2*inten/topinten_highest
    inten=[int(x) for x in inten]
    
    # save norm_factor=topinten_highest/topinten_2sum
    norm_factor=int(norm_1e4*topinten_highest/topinten_2sum)
    
    return mz,inten,norm_factor

# get_spec_df_from_pkl
def get_spec_df_from_pkl(specLib_pkl, top_ions, binsize):
    
    norm_1e2 = 1e2 # RT
    norm_1e4 = 1e4 # prec_MZ
    
    # specLib_norm_pkl
    specLib_norm_pkl = Change_ext(specLib_pkl,'_top'+str(int(top_ions))+'_per_100mz.pkl')
    if os.path.isfile(specLib_norm_pkl)==True:
        file = open(specLib_norm_pkl, "rb")
        expDF = pickle.load(file)
        file.close()
        return expDF
    
    # specLib_pkl
    if os.path.isfile(specLib_pkl)==False:
        data = {"scan": []}
        expDF = pd.DataFrame(data)
        print('file does not exist: {}'.format(specLib_pkl))
        return expDF
    
    # expDF
    file = open(specLib_pkl, "rb")
    expDF = pickle.load(file)
    file.close()
    
    # prec_MZ, RT, norm_factor
    expDF["prec_MZ"] = expDF.apply(lambda x: int(norm_1e4*x.prec_MZ)*1.0/norm_1e4, axis=1)
    expDF["RT"] = expDF.apply(lambda x: int(norm_1e2*x.RT)*1.0/norm_1e2, axis=1)
    expDF["norm_factor"] = expDF.apply(lambda x: int(x.RT), axis=1)
    
    # mz_cols1, np_arr1
    mz_cols1 = list(expDF.columns)
    np_arr1 = expDF.to_numpy()
    
    # m/z, intensity, norm_factor
    ms2_mz = []
    ms2_int = []
    ms2_normf = []
    for tno in range(0,len(np_arr1)):
        row1=np_arr1[tno]
        
        # load pkl (lib_data.columns: Index(['scan', 'charge', '[M+H]+', 'prec_MZ', 'L_ID', 'L_peptide', 'L_protein', 'RT', 'm/z', 'intensity', 'norm_factor']) )
        exp_mz_list = row1[mz_cols1.index("m/z")]
        exp_inten_list = row1[mz_cols1.index("intensity")]
        
        # picks up top 10 ions from each bin (100 mz)
        [exp_mz_list,exp_inten_list,norm_factor] = preprocess_topk_per_100mz(exp_mz_list,exp_inten_list,top_ions,binsize)
        
        # m/z, intensity
        ms2_mz.append(exp_mz_list)
        ms2_int.append(exp_inten_list)
        ms2_normf.append(norm_factor)
    
    # m/z, intensity, norm_factor
    dict1 = {"m/z":ms2_mz,"intensity":ms2_int,"norm_factor":ms2_normf}
    expDF1 = pd.DataFrame.from_dict(dict1)
    expDF["m/z"] = expDF1["m/z"]
    expDF["intensity"] = expDF1["intensity"]
    expDF["norm_factor"] = expDF1["norm_factor"]
    
    # sort expDF by prec_MZ
    expDF = expDF.sort_values(by='prec_MZ')
    
    expDF.to_pickle(specLib_norm_pkl)
    
    return expDF

# get_spec_df_from_ms2
def get_spec_df_from_ms2(exp_ms2, top_ions, binsize, sim_mass=0.0):
    
    norm_1e2 = 1e2 # RT
    norm_1e4 = 1e4 # prec_MZ
    
    if os.path.isfile(exp_ms2)==False:
        data = {"scan": []}
        expDF = pd.DataFrame(data)
        print('file does not exist: {}'.format(exp_ms2))
        return expDF
    
    g = open(exp_ms2,"r")
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
    ms2_normf = []

    mz_list = []

    for line in lines:
        if "S\t" in line:
            if len(mz_list) >= 1:
                [mz_list,int_list,norm_factor] = preprocess_topk_per_100mz(mz_list,int_list,top_ions,binsize)
                ms2_mz.append(mz_list)
                ms2_int.append(int_list)
                ms2_normf.append(norm_factor)
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
        elif "I\tRetTime" in line:# from *.ms2
    #         print ("RT found")
            temp_line = line.strip().split("\t")
            RT = float(temp_line[2])
            RT_list.append(RT)

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
        
        elif "L\tRT" in line:# from *.splib
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

    [mz_list,int_list,norm_factor] = preprocess_topk_per_100mz(mz_list,int_list,top_ions,binsize)
    ms2_mz.append(mz_list)
    ms2_int.append(int_list)
    ms2_normf.append(norm_factor)

    dict1 = {"scan":scan_list,"charge":charge_list,"[M+H]+":MH_list,
                "prec_MZ":precursorIon_list,"L_ID":L_ID_list,"L_peptide":L_peptide_list,
                "L_protein":L_Protein_list,"RT":RT_list,"m/z":ms2_mz,"intensity":ms2_int,"norm_factor":ms2_normf}

    dict2 = {}
    for key in dict1.keys():
        if len(dict1[key]) != 0:
            dict2[key] = dict1[key]

    expDF = pd.DataFrame.from_dict(dict2)
    
    # prec_MZ, RT
    expDF["prec_MZ"] = expDF.apply(lambda x: int(norm_1e4*x.prec_MZ)*1.0/norm_1e4, axis=1)
    expDF["RT"] = expDF.apply(lambda x: int(norm_1e2*x.RT)*1.0/norm_1e2, axis=1)
    
    return expDF

# get_spectra_from_df
def get_spectra_from_df(expDF,nLib):
    
    mz_cols1 = list(expDF.columns)
    np_arr1 = expDF.to_numpy()
    
    # # all_premz
    # all_premz=np.array([row[3] for row in np_arr1])
    
    # # r1: rank of all_premz by ascending order (the last is the highest)
    # ix = np.argsort(all_premz)
    # r1 = np.argsort(ix)
    
    # query_spectra, exp_spec_ID
    query_spectra = []
    exp_spec_ID = [[] for _ in range(len(np_arr1))]  # Initialize with empty lists
    
    for rno in range(0,len(np_arr1)):
        row1=np_arr1[rno]
        # tno=np.nonzero(r1==rno)[0][0]
        # row1=np_arr1[tno]
        
        # load pkl (lib_data.columns: Index(['scan', 'charge', '[M+H]+', 'prec_MZ', 'L_ID', 'L_peptide', 'L_protein', 'RT', 'm/z', 'intensity', 'norm_factor']) )
        exp_prec_mz = float( row1[mz_cols1.index("prec_MZ")] )
        exp_prech = int( row1[mz_cols1.index("charge")] )
        
        exp_mz = row1[mz_cols1.index("m/z")]
        exp_inten = row1[mz_cols1.index("intensity")]
        
        # exp_spec_ID
        if nLib==1:# library spec
            L_RT = float( row1[mz_cols1.index("RT")] )
            L_ID = row1[mz_cols1.index("L_ID")]
            L_peptide = row1[mz_cols1.index("L_peptide")]
            L_protein = row1[mz_cols1.index("L_protein")]
            exp_spec_ID[rno] = [exp_prec_mz,exp_prech,L_RT,L_ID,L_peptide,L_protein]# exp_prec_mz,exp_prech,L_RT,L_ID,L_peptide,L_protein
        else:# query spec
            cur_scan = int(row1[mz_cols1.index("scan")])
            exp_spec_ID[rno] = [exp_prec_mz,exp_prech,cur_scan]# exp_prec_mz,exp_prech,cur_scan
        
        # cur_spec
        cur_spec = {}
        cur_spec["id"] = rno
        cur_spec["precursor_mz"] = exp_prec_mz
        cur_spec["peaks"] = []
        for idx, cur_mz in enumerate(exp_mz):
            cur_spec["peaks"].append([cur_mz, exp_inten[idx]])
        
        query_spectra.append(cur_spec)
    # pprint.pprint(query_spectra)
    
    return query_spectra,exp_spec_ID

# create_fragment_index
def create_fragment_index(expDF,specLib_pkl=''):
    
    if os.path.isfile(specLib_pkl)==True:
        # frag_index_file
        frag_index_file = Change_ext(specLib_pkl,'_frag_index.dat')
        if os.path.isfile(frag_index_file)==True:
            frag_index = pickle.load(open(frag_index_file,'rb'))
            return frag_index
    
    mz_cols1 = list(expDF.columns)
    np_arr1 = expDF.to_numpy()
    
    # frag_index
    frag_index = {}
    
    for rno in range(0,len(np_arr1)):
        row1=np_arr1[rno]
        
        # load pkl (lib_data.columns: Index(['scan', 'charge', '[M+H]+', 'prec_MZ', 'L_ID', 'L_peptide', 'L_protein', 'RT', 'm/z', 'intensity', 'norm_factor']) )
        exp_prec_mz = row1[mz_cols1.index("prec_MZ")]
        exp_mz = row1[mz_cols1.index("m/z")]
        
        # frag_index
        for ino in range(0,len(exp_mz)):
            cur_frag_mz = exp_mz[ino]
            cur_prec_mz = exp_prec_mz
            if cur_frag_mz not in frag_index.keys():
                frag_index.update({cur_frag_mz:{cur_prec_mz:[rno]}})
            else:
                if cur_prec_mz not in frag_index[cur_frag_mz].keys():
                    frag_index[cur_frag_mz][cur_prec_mz]=[rno]
                else:
                    frag_index[cur_frag_mz][cur_prec_mz].append(rno)
    
    if os.path.isfile(specLib_pkl)==True:
        pickle.dump(frag_index,open(frag_index_file,'wb'))
    
    return frag_index

# get_lib_candidates
def get_lib_candidates(frag_index,all_frag,exp_prec_mz,exp_mz,ms1_tol,ms2_tol,match_threshold):
    
    # norm_1e4 = 1e4 # prec_MZ
    candidates = {}
    candidate_list = []
    
    delta_mz1 = ms1_tol*exp_prec_mz*1e-6
    left_lim1=exp_prec_mz-delta_mz1#int(norm_1e4*(exp_prec_mz-delta_mz1))*1.0/norm_1e4
    right_lim1=exp_prec_mz+delta_mz1#int(norm_1e4*(exp_prec_mz+delta_mz1))*1.0/norm_1e4
    
    for ino in range(0,len(exp_mz)):
        if ms2_tol>1.0:
            delta_mz2 = ms2_tol*exp_mz[ino]*1e-6
        else:
            delta_mz2 = ms2_tol
        left_lim2 = exp_mz[ino]-delta_mz2#int(norm_1e4*(exp_mz[ino]-delta_mz2))*1.0/norm_1e4
        right_lim2 = exp_mz[ino]+delta_mz2#int(norm_1e4*(exp_mz[ino]+delta_mz2))*1.0/norm_1e4
        
        pos2=np.nonzero((all_frag>=left_lim2) & (all_frag<=right_lim2))[0]
        if len(pos2)==0:
            continue
        
        for cur_R_frag_mz in all_frag[pos2]:
            cur_prec = np.array([x for x in frag_index[cur_R_frag_mz].keys()])
            
            pos1=np.nonzero((cur_prec>=left_lim1) & (cur_prec<=right_lim1))[0]
            if len(pos1)==0:
                continue
            
            for cur_R_prec_mz in cur_prec[pos1]:
                for jno in range(0,len(frag_index[cur_R_frag_mz][cur_R_prec_mz])):
                    cur_L_ID=frag_index[cur_R_frag_mz][cur_R_prec_mz][jno]
                    if cur_L_ID not in candidates.keys():
                        candidates.update({cur_L_ID:[1]})
                    else:
                        candidates[cur_L_ID][0]+=1
    
    for cur_L_ID in candidates.keys():
        if candidates[cur_L_ID][0]>=match_threshold:
            candidate_list.append(cur_L_ID)
    
    return candidate_list

# entropy_sim_per_peak
def entropy_sim_per_peak(x1,x2):
    # entropy similarity
    if x1==0.0 or x2==0.0:
        y = 0.0
    else:
        y = (x1+x2)*math.log((x1+x2),2) - x1*math.log(x1,2) - x2*math.log(x2,2)
    return y

# get_similarity
def get_similarity(e_intens,r_intens):
    
    if np.sqrt( sum(e_intens*e_intens)*sum(r_intens*r_intens) )==0.0:
        e_sim=0.0
    else:
        e_sim = sum(e_intens*r_intens)/np.sqrt( sum(e_intens*e_intens)*sum(r_intens*r_intens) )
    
    return e_sim

def get_pep_seq_only(seq):
    # get_pep_seq_only
    ABC='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    seq_only=''
    for i in range(0,len(seq)):
        if seq[i] in ABC:
            seq_only=seq_only+seq[i]
    
    return seq_only

def get_delta_cn_rank(psm_score):
    # get_delta_cn_rank
    nrow = np.size(psm_score,0)
    # ncol = np.size(psm_score,1)
    delta_cn=np.zeros((nrow,1))
    
    score_col = np.array([psm_score[ino][0] for ino in range(0,nrow)])
    idx = (-score_col).argsort() # descending order
    high2low_rank = np.argsort(idx)
    
    score_col.sort() # ascending order
    for ino in range(0,nrow):
        c1_score = psm_score[ino][0]
        ix = np.nonzero(score_col<c1_score)[0]
        if len(ix)==0:
            c2_score = 0.0
        else:
            c2_score = score_col[ix[-1]]
        
        if c1_score==0.0:
            delta_cn[ino][0] = 0.0
        else:
            delta_cn[ino][0] = (c1_score-c2_score)/c1_score
    
    return delta_cn,high2low_rank



def get_pep_seq_mod(c_seq_mod):
    # get_pep_seq_mod
    if c_seq_mod.find("229.162932")!=-1:
        s_n = "n[230]"
        s_M = "M[147]"
        s_C = "C[160]"
        s_K = "K[357]"
    elif c_seq_mod.find("304.2071453")!=-1:
        s_n = "n[305]"
        s_M = "M[147]"
        s_C = "C[160]"
        s_K = "K[432]"
    else:
        s_n = ""
        s_M = "M[147]"
        s_C = "C[160]"
        s_K = ""
    
    c_seq_mod = c_seq_mod.replace("15.99492+229.162932", "")
    c_seq_mod = c_seq_mod.replace("57.02146+229.162932", "")
    c_seq_mod = c_seq_mod.replace("229.162932+229.162932", "")
    c_seq_mod = c_seq_mod.replace("15.99492+304.2071453", "")
    c_seq_mod = c_seq_mod.replace("57.02146+304.2071453", "")
    c_seq_mod = c_seq_mod.replace("304.2071453+304.2071453", "")
    c_seq_mod = c_seq_mod.replace("15.99492", "")
    c_seq_mod = c_seq_mod.replace("57.02146", "")
    c_seq_mod = c_seq_mod.replace("229.162932", "")
    c_seq_mod = c_seq_mod.replace("304.2071453", "")
    c_seq_mod = c_seq_mod.replace(")", "")
    
    c_seq_mod = c_seq_mod.replace("M(", s_M)
    c_seq_mod = c_seq_mod.replace("C(", s_C)
    c_seq_mod = c_seq_mod.replace("K(", s_K)
    c_seq_mod = c_seq_mod.replace("(", "")
    
    n_seq_mod = s_n + c_seq_mod
    
    return n_seq_mod

def get_seq_only_and_seq_mod(c_peptide):
    # get_seq_only_and_seq_mod
    
    if "Decoy_" in c_peptide:
        c_seq_only = get_pep_seq_only(c_peptide.split("_")[1])
        c_seq_mod = c_peptide.split("_")[1]
    else:
        c_seq_only = get_pep_seq_only(c_peptide)
        c_seq_mod = c_peptide
    c_seq_mod = get_pep_seq_mod(c_seq_mod)
    
    return c_seq_only, c_seq_mod

def get_mod_pos_mass(n_seq_mod):
    # get_mod_pos_mass
    
    dict1 = {'n[230]': 'a', 'n[305]': 'b', 'M[147]': 'c', 'C[160]': 'd', 'K[357]': 'e', 'K[432]': 'f'}
    dict2 = {'a': '230.170757032', 'b': '305.214970332', 'c': '147.03540491299', 'd': '160.03064478471', 'e': '357.257895014', 'f': '432.302108314'}
    
    n_seq_mod = n_seq_mod.replace('n[230]', "a")
    n_seq_mod = n_seq_mod.replace('n[305]', "b")
    n_seq_mod = n_seq_mod.replace('M[147]', "c")
    n_seq_mod = n_seq_mod.replace('C[160]', "d")
    n_seq_mod = n_seq_mod.replace('K[357]', "e")
    n_seq_mod = n_seq_mod.replace('K[432]', "f")
    
    mod_pos = []
    mod_mass = []
    
    nshift = 1
    if n_seq_mod[0]=='a' or n_seq_mod[0]=='b':
        nshift = 0
    
    ABC='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(0,len(n_seq_mod)):
        if n_seq_mod[i] not in ABC:
            mod_pos.append( str(i+nshift) )
            mod_mass.append( dict2[n_seq_mod[i]] )
    
    return mod_pos, mod_mass

def Find_AllSubStr(test_str,test_sub):
    # find all sub string and return all index
    index = []
    if len(test_str)<len(test_sub):
        return index
    for i in range(0,len(test_str)):
        #if test_str.startswith(test_sub,i,i+len(test_sub)-1):
        if test_str.startswith(test_sub,i,i+len(test_sub)):
            index.append(i)
    return index

def get_num_missed_cleavages(c_seq_only):
    # get_num_missed_cleavages
    
    c_num_missed_cleavages = 0
    
    idx_K = Find_AllSubStr(c_seq_only,'K')
    idx_R = Find_AllSubStr(c_seq_only,'R')
    idx_KP = Find_AllSubStr(c_seq_only,'KP')
    idx_RP = Find_AllSubStr(c_seq_only,'RP')
    
    ctermK = 0
    if c_seq_only[-1]=='K':
        ctermK = 1
    
    ctermR = 0
    if c_seq_only[-1]=='R':
        ctermR = 1
    
    c_num_missed_cleavages = len(idx_K)-len(idx_KP)-ctermK + len(idx_R)-len(idx_RP)-ctermR
    
    return c_num_missed_cleavages

def get_godel(c_str):
    # get_godel
    pre_nums = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987,1993,1997,1999,2003,2011,2017,2027,2029,2039,2053,2063,2069,2081,2083,2087,2089,2099,2111,2113,2129,2131,2137,2141,2143,2153,2161,2179,2203,2207,2213,2221,2237,2239,2243,2251,2267,2269,2273,2281,2287,2293,2297,2309,2311,2333,2339,2341,2347,2351,2357,2371,2377,2381,2383,2389,2393,2399,2411,2417,2423,2437,2441,2447,2459,2467,2473,2477,2503,2521,2531,2539,2543,2549,2551,2557,2579,2591,2593,2609,2617,2621,2633,2647,2657,2659,2663,2671,2677,2683,2687,2689,2693,2699,2707,2711,2713,2719,2729,2731,2741,2749,2753,2767,2777,2789,2791,2797,2801,2803,2819,2833,2837,2843,2851,2857,2861,2879,2887,2897,2903,2909,2917,2927,2939,2953,2957,2963,2969,2971,2999,3001,3011,3019,3023,3037,3041,3049,3061,3067,3079,3083,3089,3109,3119,3121,3137,3163,3167,3169,3181,3187,3191,3203,3209,3217,3221,3229,3251,3253,3257,3259,3271,3299,3301,3307,3313,3319,3323,3329,3331,3343,3347,3359,3361,3371,3373,3389,3391,3407,3413,3433,3449,3457,3461,3463,3467,3469,3491,3499,3511,3517,3527,3529,3533,3539,3541,3547,3557,3559,3571,3581,3583,3593,3607,3613,3617,3623,3631,3637,3643,3659,3671,3673,3677,3691,3697,3701,3709,3719,3727,3733,3739,3761,3767,3769,3779,3793,3797,3803,3821,3823,3833,3847,3851,3853,3863,3877,3881,3889,3907,3911,3917,3919,3923,3929,3931,3943,3947,3967,3989,4001,4003,4007,4013,4019,4021,4027,4049,4051,4057,4073,4079,4091,4093,4099,4111,4127,4129,4133,4139,4153,4157,4159,4177,4201,4211,4217,4219,4229,4231,4241,4243,4253,4259,4261,4271,4273,4283,4289,4297,4327,4337,4339,4349,4357,4363,4373,4391,4397,4409,4421,4423,4441,4447,4451,4457,4463,4481,4483,4493,4507,4513,4517,4519,4523,4547,4549,4561,4567,4583,4591,4597,4603,4621,4637,4639,4643,4649,4651,4657,4663,4673,4679,4691,4703,4721,4723,4729,4733,4751,4759,4783,4787,4789,4793,4799,4801,4813,4817,4831,4861,4871,4877,4889,4903,4909,4919,4931,4933,4937,4943,4951,4957,4967,4969,4973,4987,4993,4999,5003,5009,5011,5021,5023,5039,5051,5059,5077,5081,5087,5099,5101,5107,5113,5119,5147,5153,5167,5171,5179,5189,5197,5209,5227,5231,5233,5237,5261,5273,5279,5281,5297,5303,5309,5323,5333,5347,5351,5381,5387,5393,5399,5407,5413,5417,5419,5431,5437,5441,5443,5449,5471,5477,5479,5483,5501,5503,5507,5519,5521,5527,5531,5557,5563,5569,5573,5581,5591,5623,5639,5641,5647,5651,5653,5657,5659,5669,5683,5689,5693,5701,5711,5717,5737,5741,5743,5749,5779,5783,5791,5801,5807,5813,5821,5827,5839,5843,5849,5851,5857,5861,5867,5869,5879,5881,5897,5903,5923,5927,5939,5953,5981,5987,6007,6011,6029,6037,6043,6047,6053,6067,6073,6079,6089,6091,6101,6113,6121,6131,6133,6143,6151,6163,6173,6197,6199,6203,6211,6217,6221,6229,6247,6257,6263,6269,6271,6277,6287,6299,6301,6311,6317,6323,6329,6337,6343,6353,6359,6361,6367,6373,6379,6389,6397,6421,6427,6449,6451,6469,6473,6481,6491,6521,6529,6547,6551,6553,6563,6569,6571,6577,6581,6599,6607,6619,6637,6653,6659,6661,6673,6679,6689,6691,6701,6703,6709,6719,6733,6737,6761,6763,6779,6781,6791,6793,6803,6823,6827,6829,6833,6841,6857,6863,6869,6871,6883,6899,6907,6911,6917,6947,6949,6959,6961,6967,6971,6977,6983,6991,6997,7001,7013,7019,7027,7039,7043,7057,7069,7079,7103,7109,7121,7127,7129,7151,7159,7177,7187,7193,7207,7211,7213,7219,7229,7237,7243,7247,7253,7283,7297,7307,7309,7321,7331,7333,7349,7351,7369,7393,7411,7417,7433,7451,7457,7459,7477,7481,7487,7489,7499,7507,7517,7523,7529,7537,7541,7547,7549,7559,7561,7573,7577,7583,7589,7591,7603,7607,7621,7639,7643,7649,7669,7673,7681,7687,7691,7699,7703,7717,7723,7727,7741,7753,7757,7759,7789,7793,7817,7823,7829,7841,7853,7867,7873,7877,7879,7883,7901,7907,7919,7927,7933,7937,7949,7951,7963,7993,8009,8011,8017,8039,8053,8059,8069,8081,8087,8089,8093,8101,8111,8117,8123,8147,8161,8167,8171,8179,8191,8209,8219,8221,8231,8233,8237,8243,8263,8269,8273,8287,8291,8293,8297,8311,8317,8329,8353,8363,8369,8377,8387,8389,8419,8423,8429,8431,8443,8447,8461,8467,8501,8513,8521,8527,8537,8539,8543,8563,8573,8581,8597,8599,8609,8623,8627,8629,8641,8647,8663,8669,8677,8681,8689,8693,8699,8707,8713,8719,8731,8737,8741,8747,8753,8761,8779,8783,8803,8807,8819,8821,8831,8837,8839,8849,8861,8863,8867,8887,8893,8923,8929,8933,8941,8951,8963,8969,8971,8999,9001,9007,9011,9013,9029,9041,9043,9049,9059,9067,9091,9103,9109,9127,9133,9137,9151,9157,9161,9173,9181,9187,9199,9203,9209,9221,9227,9239,9241,9257,9277,9281,9283,9293,9311,9319,9323,9337,9341,9343,9349,9371,9377,9391,9397,9403,9413,9419,9421,9431,9433,9437,9439,9461,9463,9467,9473,9479,9491,9497,9511,9521,9533,9539,9547,9551,9587,9601,9613,9619,9623,9629,9631,9643,9649,9661,9677,9679,9689,9697,9719,9721,9733,9739,9743,9749,9767,9769,9781,9787,9791,9803,9811,9817,9829,9833,9839,9851,9857,9859,9871,9883,9887,9901,9907,9923,9929,9931,9941,9949,9967,9973]
    
    godel_code=0
    if len(c_str)<=len(pre_nums) and len(c_str)>0:
        for i in range(0,len(c_str)):
            c_code = (ord(c_str[i])+1)*math.log(pre_nums[i])
            godel_code=godel_code+c_code
        godel_code=int(godel_code*1e10)
    else:
        pass
        #print('%d>%d' % (len(c_str),len(pre_nums)))
    
    return godel_code

def Convert_JUMPlib_csv2pepXML(c_csv_fullfile):
    # Load the CSV file
    csv_file_path = c_csv_fullfile
    data0 = pd.read_csv(csv_file_path)
    data0["spectrum_code"] = data0.apply(lambda x: get_godel( os.path.split( x["Outfile"] )[1] ), axis=1)# run.scan.no.ch
    data0["hit_rank"] = data0.apply(lambda x: int( x["Rank (PSMS)"][4:] ), axis=1)
    
    # Sort by the scan number, spectrum_code, hit_rank
    data = data0.sort_values(by=['scan', 'spectrum_code', 'hit_rank'], ascending=[True, True, True])
    
    # Get current date
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Replace NaN values with an empty string
    data = data.fillna("")
    
    # Create the root element for pepXML
    root = ET.Element("msms_pipeline_analysis", {
        "date": dt_string,
        "summary_xml": os.path.split( Change_ext(c_csv_fullfile, '.pep.xml') )[1]
    })
    
    # Create a sample element
    msms_run_summary = ET.SubElement(root, "msms_run_summary", {
        "base_name": os.path.split( c_csv_fullfile )[1].split('.')[0],
        "raw_data_type": "raw",
        "raw_data": ".mzXML"
    })
    
    # Add sample_enzyme
    sample_enzyme = ET.SubElement(msms_run_summary, "sample_enzyme", {
        "name": "Trypsin"
    })
    
    # Add specificity
    specificity = ET.SubElement(sample_enzyme, "specificity", {
        "cut": "KR",
        "no_cut": "P",
        "sense": "C"
    })
    
    # Add search_summary
    search_summary = ET.SubElement(msms_run_summary, "search_summary", {
        "base_name": os.path.split( c_csv_fullfile )[1].split('.')[0],
        "search_engine": "JUMPlib",
        "precursor_mass_type": "monoisotopic",
        "fragment_mass_type": "monoisotopic",
        "search_id": "1"
    })
    
    # Add parameter
    parameter = ET.SubElement(search_summary, "parameter", {
        "name": "ion_series",
        "value": "1 1 0 0 0 0 0 1 0"
    })
    
    # Iterate over rows and create XML structure
    idx_no = 0
    old_spectrum = 0
    for _, row in data.iterrows():
        # Start a new spectrum
        cur_spectrum = int( row['spectrum_code'] )
        if old_spectrum!=cur_spectrum:
            idx_no = idx_no+1
            old_spectrum = cur_spectrum
            
            # Add spectrum_query
            spectrum_query = ET.SubElement(msms_run_summary, "spectrum_query", {
                "spectrum": os.path.split( str(row['Outfile']) )[1],
                "start_scan": str(row['scan']),
                "end_scan": str(row['scan']),
                "precursor_neutral_mass": str( float(row['measuredMH'])-1.007276 ),
                "assumed_charge": str(row['Outfile']).split('.')[3],
                "index": str(idx_no)
            })
            
            # Add search_result
            search_result = ET.SubElement(spectrum_query, "search_result")
        
        # Get c_seq_only, c_seq_mod, c_proteins, c_num_missed_cleavages, c_mod_pos, c_mod_mass
        [c_seq_only, c_seq_mod] = get_seq_only_and_seq_mod( str(row['Peptide']) )
        c_proteins = str(row['Protein']).split(',')
        c_num_missed_cleavages = get_num_missed_cleavages(c_seq_only)
        [c_mod_pos, c_mod_mass] = get_mod_pos_mass(c_seq_mod)
        
        # Work through all search_hit
        # Add search_hit
        search_hit = ET.SubElement(search_result, "search_hit", {
            "hit_rank": str(row['Rank (PSMS)'])[4:],
            "peptide": c_seq_only,
            "peptide_prev_aa": str(row['pep_prev_aa']),
            "peptide_next_aa": str(row['pep_next_aa']),
            "protein": c_proteins[0],
            "num_tot_proteins": str(len(c_proteins)),
            "num_matched_ions": str(row['num_matched_ions']),
            "tot_num_ions": str(row['tot_num_ions']),
            "calc_neutral_pep_mass": str( float(row['calcMH'])-1.007276 ),
            "massdiff": str(row['ppm']),
            "num_tol_term": "2",
            "num_missed_cleavages": str(c_num_missed_cleavages),
            "is_rejected": "0"
        })
        
        if len(c_proteins)>1:
            for pno1 in range(1, len(c_proteins)):
                # Add alternative_protein
                alternative_protein = ET.SubElement(search_hit, "alternative_protein", {
                    "protein": c_proteins[pno1]
                })
        
        if len(c_mod_pos)>0:
            # Add modification_info
            if c_mod_pos[0]=='0':
                modification_info = ET.SubElement(search_hit, "modification_info", {
                    "mod_nterm_mass": c_mod_mass[0],
                    "modified_peptide": c_seq_mod
                })
                nstart = 1
            else:
                modification_info = ET.SubElement(search_hit, "modification_info", {
                    "modified_peptide": c_seq_mod
                })
                nstart = 0
            
            for pno2 in range(nstart, len(c_mod_pos)):
                # Add mod_aminoacid_mass
                mod_aminoacid_mass = ET.SubElement(modification_info, "mod_aminoacid_mass", {
                    "position": c_mod_pos[pno2],
                    "mass": c_mod_mass[pno2]
                })
        
        # Add search_score
        search_score = ET.SubElement(search_hit, "search_score", {
            "name": "JDscore",
            "value": str(row['JDscore'])
        })
                
        # Add search_score
        search_score = ET.SubElement(search_hit, "search_score", {
            "name": "deltacn",
            "value": str(row['deltacn'])
        })
    
    # Convert the XML structure to a string with encoding
    xml_str = ET.tostring(root, encoding="UTF-8")
    
    # Prettify the XML string
    xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ", encoding="UTF-8")
    
    # Save the prettified XML string to a file
    xml_file_path = Change_ext(c_csv_fullfile, '.pep.xml')
    with open(xml_file_path, 'wb') as f:  # Use 'wb' to write bytes
        f.write(xml_str)
    
    return
