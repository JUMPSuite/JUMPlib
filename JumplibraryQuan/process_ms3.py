import re
import numpy as np

def Get_ms123scan_list_via_mzXML(reader):
    # Get_ms123scan_list
    ms1scan_old = 0
    ms1scan_list_cur = []
    ms2scan_list_cur = []
    ms2premz_list_cur = []
    ms123scan_list = []
    
    for ino in range(0,len(reader)):
        spec = reader[ino]
        msLevel = int(spec["msLevel"])
        msScan = int(spec["num"])
        if msLevel==1:
            # MS1
            ms1scan = msScan
            if ms1scan_old!=ms1scan:
                ms1scan_list_pre = ms1scan_list_cur
                ms2scan_list_pre = ms2scan_list_cur
                ms2premz_list_pre = ms2premz_list_cur
                
                ms1scan_old = ms1scan
                ms1scan_list_cur = []
                ms2scan_list_cur = []
                ms2premz_list_cur = []
        
        if msLevel==2:
            # MS2
            ms2scan = msScan
            try:
                filterLine = spec["filterLine"] # converted from ReAdW
            except:
                filterLine = "-" # converted from msconvert
            if filterLine != "-":
                pre_mz = float(re.search(r"ms2 ([0-9.]+)\@", spec["filterLine"]).group(1)) # converted from ReAdW
            else:
                pre_mz = int(float(spec["precursorMz"][-1]["precursorMz"])*1e4+0.4)/1e4 # converted from msconvert
            
            ms1scan_list_cur.append(ms1scan)
            ms2scan_list_cur.append(ms2scan)
            ms2premz_list_cur.append(pre_mz)
        
        if msLevel==3:
            # MS3
            ms3scan = msScan
            try:
                filterLine = spec["filterLine"] # converted from ReAdW
            except:
                filterLine = "-" # converted from msconvert
            if filterLine != "-":
                pre_mz = float(re.search(r"ms3 ([0-9.]+)\@", spec["filterLine"]).group(1)) # converted from ReAdW
            else:
                pre_mz = int(float(spec["precursorMz"][-1]["precursorMz"])*1e4+0.4)/1e4 # converted from msconvert
            
            ms1scan_list = ms1scan_list_pre + ms1scan_list_cur
            ms2scan_list = ms2scan_list_pre + ms2scan_list_cur
            ms2premz_list = ms2premz_list_pre + ms2premz_list_cur
            ms2premz_array = np.array(ms2premz_list)
            pos1=np.nonzero((ms2premz_array>=pre_mz-0.0001) & (ms2premz_array<=pre_mz+0.0001))[0]
            if len(pos1)>0:
                ms123scan_list.append( (ms1scan_list[pos1[0]],ms2scan_list[pos1[0]],ms3scan) )
    
    ms123scan_ms2_list = []
    ms123scan_ms3_list = []
    for c_ms123scan in ms123scan_list:
        ms123scan_ms2_list.append(c_ms123scan[1])
        ms123scan_ms3_list.append(c_ms123scan[2])
    ms123scan_ms2_array = np.array(ms123scan_ms2_list)
    
    return ms123scan_list, ms123scan_ms2_array, ms123scan_ms3_list

def Get_ms123scan_list_via_mzML(reader):
    # Get_ms123scan_list
    ms1scan_old = 0
    ms1scan_list_cur = []
    ms2scan_list_cur = []
    ms2premz_list_cur = []
    ms123scan_list = []
    
    for ino in range(0,len(reader)):
        spec = reader[ino]
        msLevel = spec['ms level']
        msScan = int(spec['id'].split('scan=')[-1])
        if msLevel==1:
            # MS1
            ms1scan = msScan
            if ms1scan_old!=ms1scan:
                ms1scan_list_pre = ms1scan_list_cur
                ms2scan_list_pre = ms2scan_list_cur
                ms2premz_list_pre = ms2premz_list_cur
                
                ms1scan_old = ms1scan
                ms1scan_list_cur = []
                ms2scan_list_cur = []
                ms2premz_list_cur = []
        
        if msLevel==2:
            # MS2
            ms2scan = msScan
            try:
                pre_mz = float(re.search(r"ms2 ([0-9.]+)\@", spec['scanList']['scan'][0]['filter string']).group(1))
            except:
                pre_mz = int(spec['precursorList']['precursor'][-1]['selectedIonList']['selectedIon'][0]['selected ion m/z']*1e4+0.4)/1e4
            
            ms1scan_list_cur.append(ms1scan)
            ms2scan_list_cur.append(ms2scan)
            ms2premz_list_cur.append(pre_mz)
        
        if msLevel==3:
            # MS3
            ms3scan = msScan
            try:
                pre_mz = float(re.search(r"ms3 ([0-9.]+)\@", spec['scanList']['scan'][0]['filter string']).group(1))
            except:
                pre_mz = int(spec['precursorList']['precursor'][-1]['selectedIonList']['selectedIon'][0]['selected ion m/z']*1e4+0.4)/1e4
            
            ms1scan_list = ms1scan_list_pre + ms1scan_list_cur
            ms2scan_list = ms2scan_list_pre + ms2scan_list_cur
            ms2premz_list = ms2premz_list_pre + ms2premz_list_cur
            ms2premz_array = np.array(ms2premz_list)
            pos1=np.nonzero((ms2premz_array>=pre_mz-0.0001) & (ms2premz_array<=pre_mz+0.0001))[0]
            if len(pos1)>0:
                ms123scan_list.append( (ms1scan_list[pos1[0]],ms2scan_list[pos1[0]],ms3scan) )
    
    ms123scan_ms2_list = []
    ms123scan_ms3_list = []
    for c_ms123scan in ms123scan_list:
        ms123scan_ms2_list.append(c_ms123scan[1])
        ms123scan_ms3_list.append(c_ms123scan[2])
    ms123scan_ms2_array = np.array(ms123scan_ms2_list)
    
    return ms123scan_list, ms123scan_ms2_array, ms123scan_ms3_list

def Merge_ms23(ms2mz,ms2inten,ms3mz,ms3inten):
    # Merge_ms23
    
    # Cut ms2mz and ms2inten
    cutoff_ms2 = 136
    left_ms2_indices = np.where(ms2mz > cutoff_ms2)
    left_ms2mz = ms2mz[left_ms2_indices]
    left_ms2inten = ms2inten[left_ms2_indices]
    
    # Cut ms3mz and ms3inten
    cutoff_ms3 = 136
    left_ms3_indices = np.where(ms3mz < cutoff_ms3)
    left_ms3mz = ms3mz[left_ms3_indices]
    left_ms3inten = ms3inten[left_ms3_indices]
    
    # Merge the left of ms2 and ms3
    ms23mz = np.concatenate((left_ms3mz, left_ms2mz))
    ms23inten = np.concatenate((left_ms3inten, left_ms2inten))
    
    return ms23mz,ms23inten
