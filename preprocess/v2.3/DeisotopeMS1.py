#!/usr/bin/python

import os
import sys
import time
import re
import platform
import warnings
import shutil
import glob
import argparse
from datetime import datetime
from pyteomics import mzxml,mzml,mass
from numpy import *
from DeisotopeMS1Mono import *
from tmtCorrection_mzXML import *
from multiprocessing import Process

def Centrehist(Y,X):
	# get histogram
	Y=array(Y)
	frequence=len(X)*[0]
	for i in range(0,len(X)):
		if i==0:
			IX=nonzero(Y<=0.5*X[i+1]+0.5*X[i])[0]
		elif i==len(X)-1:
			IX=nonzero(Y>0.5*X[i-1]+0.5*X[i])[0]
		else:
			IX=nonzero((Y<=0.5*X[i+1]+0.5*X[i])&(Y>0.5*X[i-1]+0.5*X[i]))[0]
		frequence[i]=len(IX)
	return frequence

def Getbaseline(inten):
	# get baseline
	loginten=log10(inten)
	loginten=array(loginten)
	t=arange(loginten.min(),loginten.max(),0.08)
	n=Centrehist(loginten,t)
	baseline=10**t[argmax(n)]
	
	return baseline

def Get_ntype(ms2type_s,activation_s):
	# get different types with nos
	if ms2type_s=="ITMS":
		ms2type = 1
	else:# FTMS
		ms2type = 2
	
	if activation_s=="CID":
		activation = 1
	elif activation_s=="ETD":
		activation = 2
	else: # HCD
		activation = 3
	
	fragtype = (activation-1)*2+ms2type # {'CIDIT','CIDFT','ETDIT','ETDFT','HCDIT','HCDFT'}
	
	return ms2type, activation, fragtype

def get_mass_error_by_bin(MS_MassError,MS1_index,MS2_index,tol_ppm,thresholdPercentage):
	# get mass error by (10) bin
	
	num_MS1=MS1_index.shape[0]
	num_MS2=MS2_index.shape[0]
	
	if num_MS2>10000:
		binsize = 10
	else:
		binsize = 1
	
	pos=nonzero(MS_MassError<=tol_ppm)[0]
	measuredPercentage=len(pos)/len(MS_MassError)
	scan_ppm = []
	if measuredPercentage >= thresholdPercentage:
		scans = []
		scans.append(0)
		for i in range(1,binsize+1):
			cur_pos=get_cur_MS2POS(len(pos),i,binsize)
			if len(MS_MassError)==num_MS1:
				cur_scan=MS1_index[pos[cur_pos[-1]],0]
			else:
				cur_scan=MS2_index[pos[cur_pos[-1]],2]
			if i==binsize:
				cur_scan=max([MS1_index[num_MS1-1,0],MS2_index[num_MS2-1,2]])
			cur_mean=mean(MS_MassError[pos[cur_pos]])
			scan_ppm.append((cur_scan,cur_mean))
			scans.append(cur_scan)
		for ino in range(0,num_MS1):
			c_scan = MS1_index[ino,0]
			for j in range(0,binsize):
				if c_scan>scans[j] and c_scan<=scans[j+1]:
					break;
			MS1_index[ino,4]=scan_ppm[j][1]
		for ino in range(0,num_MS2):
			c_scan = MS2_index[ino,2]
			for j in range(0,binsize):
				if c_scan>scans[j] and c_scan<=scans[j+1]:
					break;
			MS2_index[ino,7]=scan_ppm[j][1]
		# print(scan_ppm)
	
	return [MS1_index,MS2_index,scan_ppm]

def ReadmzXML(t_mzXML_fullfile,raw_data_type,tmt,nrunning_mode):
	# Read mzXML
	nMS1=0
	nMS2=0
	scan_ppm=[]
	
	# check extracted MS files in 'datapath/filename'
	MS1_scanfile = Change_ext(t_mzXML_fullfile,'_MS1scans.npz')
	MS1_peakfile = Change_ext(t_mzXML_fullfile,'_MS1peaks.npz')
	MS2_scanfile = Change_ext(t_mzXML_fullfile,'_MS2scans.npz')
	MS2_peakfile = Change_ext(t_mzXML_fullfile,'_MS2peaks.npz')
	if os.path.isfile(MS1_scanfile) and os.path.isfile(MS1_peakfile) and os.path.isfile(MS2_scanfile) and os.path.isfile(MS2_peakfile):
		M1scan=load(MS1_scanfile)
		M2scan=load(MS2_scanfile)
		MS1_index=M1scan['MS1_index']
		MS2_index=M2scan['MS2_index']
		nMS1=MS1_index.shape[0]
		nMS2=MS2_index.shape[0]
		scan_ppm=M2scan['scan_ppm']
		return [nMS1,nMS2,scan_ppm]
	
	# check the mzXML file
	if os.path.isfile(t_mzXML_fullfile)==False:
		print('%s: does not exist!' % (t_mzXML_fullfile))
		return [nMS1,nMS2,scan_ppm]
	
	# mzFileToNumpyArr
	[dfmzXML,np_arr1,np_arr2] = mzFileToNumpyArr(t_mzXML_fullfile,raw_data_type)
	
	# mass correction
	thresholdPercentage = 0.05
	nTMT = len(tmt)
	if nTMT==0:
		referenceMasses = [445.1200245337]
		tol_ppm = 50
		[MS1_index,MS2_index,MS1_MassError] = MS1MassCorrection(np_arr1,np_arr2,referenceMasses,tol_ppm)
		[MS1_index,MS2_index,scan_ppm]=get_mass_error_by_bin(MS1_MassError,MS1_index,MS2_index,tol_ppm,thresholdPercentage)
	else:
		tol_ppm = 50
		[MS1_index,MS2_index,MS2_MassError] = MS2MassCorrection(np_arr1,np_arr2,tmt,tol_ppm)
		if MS2_index.shape[0]<10000:
			tol_ppm = 20
			[MS1_index,MS2_index,MS2_MassError] = MS2MassCorrection(np_arr1,np_arr2,tmt,tol_ppm)
		
		[MS1_index,MS2_index,scan_ppm]=get_mass_error_by_bin(MS2_MassError,MS1_index,MS2_index,tol_ppm,thresholdPercentage)
	
	nMS1=MS1_index.shape[0]
	nMS2=MS2_index.shape[0]
	
	# initial
	TotalPeakNum=int(7e7) # initial total peak number on MS1/MS2 scans
	#MS1_index=zeros((nMS1,5)) # MS1 scan, MS1 rt, MS1 peak num, baseline, massshift
	MS1_peaks=zeros((TotalPeakNum,2)) # m/z and intensity on MS1
	#MS2_index=zeros((nMS2,8)) # MS1 scan,MS1 rt,MS2 scan,m/z,z,Fragtype,MS2 peak num,massshift
	MS2_peaks=zeros((TotalPeakNum,2)) # m/z and intensity on MS2
	ms1_fno = 0 # real MS1 scan number
	ms1_pno = 0 # real total peak number on MS1
	ms2_fno = 0 # real MS2 scan number
	ms2_pno = 0 # real total peak number on MS2
	
	# spectra
	#print('total scans:')
	mz_cols0 = list(dfmzXML.columns)
	np_arr0 = dfmzXML.to_numpy()
	for tno in range(0,len(np_arr0)):
		row0=np_arr0[tno]
		num = row0[mz_cols0.index("num")]
		msLevel = row0[mz_cols0.index("msLevel")]
		peaksCount = row0[mz_cols0.index("peaksCount")]
		retentionTime = row0[mz_cols0.index("retentionTime")]
		msType = row0[mz_cols0.index("msType")]
		activationMethod = row0[mz_cols0.index("activationMethod")]
		precursorMz = row0[mz_cols0.index("precursorMz")]
		precursorCh = row0[mz_cols0.index("precursorCh")]
		mz_array = array(row0[mz_cols0.index("m/z array")])
		intensity_array = array(row0[mz_cols0.index("intensity array")])
		
		msScan = num
		if msLevel==1:
			# MS1
			ms1scan = msScan
			ms1npk = peaksCount
			ms1rt = retentionTime # min
			# ms1type_s = msType
			ms1mz = mz_array
			ms1inten = intensity_array
			if ms1npk>200:
				baseline = Getbaseline(ms1inten)/2.0
				IX=nonzero(ms1inten>=min(baseline,500.0))[0]
				ms1npk=len(IX)
				ms1mz=ms1mz[IX]
				ms1inten=ms1inten[IX]
			else:
				baseline = 0.0
			# # correct ms1
			# cur_massshift=MS1_index[ms1_fno,4]
			# if cur_massshift!=0:
				# ms1mz = massCorrectionFunction(ms1mz,cur_massshift)
			
			MS1_index[ms1_fno,0:4]=[ms1scan,ms1rt,ms1npk,baseline]# MS1_index[ms1_fno,4]: massshift
			#print(MS1_index[ms1_fno,0:5])
			ms1_fno = ms1_fno+1
			MS1_peaks[ms1_pno:ms1_pno+ms1npk,0]=ms1mz
			MS1_peaks[ms1_pno:ms1_pno+ms1npk,1]=ms1inten
			ms1_pno = ms1_pno+ms1npk
		if msLevel==2:
			# MS2
			ms2scan = msScan
			ms2npk = peaksCount
			ms2rt = retentionTime # min
			ms2type_s = msType
			activation_s = activationMethod
			pre_mz = precursorMz
			pre_ch = precursorCh
			ms2mz = mz_array
			ms2inten = intensity_array
			[ms2type, activation, fragtype] = Get_ntype(ms2type_s,activation_s)
			# baseline = 0.0
			
			if nrunning_mode!=4:# non-debug mode
				MS2_index[ms2_fno,0:7]=[ms1scan,ms1rt,ms2scan,pre_mz,pre_ch,fragtype,ms2npk]# MS2_index[ms2_fno,7]: massshift
			else:# nrunning_mode=4, debug mode
				MS2_index[ms2_fno,0:7]=[ms1scan,ms2rt,ms2scan,pre_mz,pre_ch,fragtype,ms2npk]# MS2_index[ms2_fno,7]: massshift
			#print(MS2_index[ms2_fno,0:8])
			ms2_fno = ms2_fno+1
			MS2_peaks[ms2_pno:ms2_pno+ms2npk,0]=ms2mz
			MS2_peaks[ms2_pno:ms2_pno+ms2npk,1]=ms2inten
			ms2_pno = ms2_pno+ms2npk
		#print(msScan, end="\r")
	#print('')
	
	#save the MS1 info
	transM=transpose(MS1_index)
	tmp=transM[2]
	tmp=array(tmp)
	transM[2]=tmp.cumsum()+1
	MS1_index=transpose(transM)
	#print(MS1_index)
	
	if ms1_pno<TotalPeakNum:
		MS1_peaks=MS1_peaks[0:ms1_pno,:]
	savez(MS1_scanfile,MS1_index=MS1_index)
	savez(MS1_peakfile,MS1_peaks=MS1_peaks)
	
	#save the MS2 info
	transM=transpose(MS2_index)
	tmp=transM[6]
	tmp=array(tmp)
	transM[6]=tmp.cumsum()+1
	MS2_index=transpose(transM)
	
	#print(MS2_index)
	
	if ms2_pno<TotalPeakNum:
		MS2_peaks=MS2_peaks[0:ms2_pno,:]
	savez(MS2_scanfile,MS2_index=MS2_index,scan_ppm=scan_ppm)
	savez(MS2_peakfile,MS2_peaks=MS2_peaks)
	
	return [nMS1,nMS2,scan_ppm]

def submit_job(jobf,queue,mem):
	# submit a job in HPC
	cmd = 'bsub -P Proteomics -q '+queue+' -R "rusage[mem='+mem+']" < '+jobf
	os.system(cmd)

def Get_hostname(update_path):
	# get hostname
	hostname_fullfile = os.path.join(update_path,'hostname.txt')
	os.system('hostname > %s' % hostname_fullfile)
	file1 = open(hostname_fullfile,'r')
	c_str = file1.readline()
	line = c_str.strip()
	file1.close()
	os.remove(hostname_fullfile)
	if line.find('splprhpc')!=-1 or line.find('node')!=-1:
		c_cluster = 'hpc'
	else:
		c_cluster = 'spiderscluster'
	return c_cluster

def GetMonoInBatch(t_mzXML_fullfile,raw_data_type,jump_params,scan_ppm,nrunning_mode):
	# Get Mono Peaks In Batch
	
	# check extracted MS files in 'datapath/filename'
	[resultpath,tname] = os.path.split(t_mzXML_fullfile)
	
	c_mzXML_fullfile = resultpath+'.{}'.format(raw_data_type)
	params = Get_params(jump_params)
	nprocessor=params['nprocessor']# no of multiprocessing nodes
	parallel_method=params['parallel_method']# parallel method, 1: multiprocessing package, 2: submitting LSF jobs
	if len(scan_ppm)==0:# 0: uncorrected; 1: corrected
		ncorrect = 0
	else:
		ncorrect = 1
	
	# hpc_node
	hpc_node = 0
	if 'Linux'==platform.system() and 'hpc'==Get_hostname(resultpath):
		hpc_node = 1
	
	if parallel_method==2 and hpc_node==1:
		# for each batch
		codepath = os.getcwd()
		queue='standard'
		mem='%d' % (12*1024)
		
		batch_py = os.path.join(codepath,'DeisotopeMS1Batch.py')
		log_path = os.path.join(resultpath,'log')
		Check_Path(log_path)
		
		for ino in range(0,nprocessor):
			# params_file
			params_file = Change_ext(t_mzXML_fullfile,'_batch%d.deiso.params' % (ino+1))
			fp = open(params_file,'w')
			fp.write('[Batch]\n')
			fp.write('datapath=%s\n' % resultpath)
			fp.write('jump_params=%s\n' % jump_params)
			fp.write('current_batch=%d\n' % (ino+1))
			fp.write('total_batch=%d\n' % nprocessor)
			fp.write('ncorrect=%d\n' % ncorrect)
			fp.close()
			
			# bat_file
			bat_file = Change_ext(t_mzXML_fullfile,'_batch%d.deiso.sh' % (ino+1))
			fid = open(bat_file,'w')
			job_header = "#!/bin/bash\n#BSUB -oo %s/log%d.out\n#BSUB -eo %s/error%d.err\n" % (log_path,ino+1,log_path,ino+1)
			fid.write('%s' % job_header)
			fid.write('python %s %s\n' % (batch_py,params_file))
			fid.close()
			
			# mono_file
			mono_file = Change_ext(t_mzXML_fullfile,'_Mono%d.npz' % (ino+1))
			if False==os.path.isfile(mono_file):
				submit_job(bat_file,queue,mem)
	else:
		# processes
		processes = []
		
		# create processes
		for cur_nbatch in range(1,nprocessor+1):
			# get monoisotopics on current batch
			p = Process(target=GetMonoPeaks, args=(c_mzXML_fullfile,jump_params,cur_nbatch,nprocessor,ncorrect))
			processes.append(p)
			p.start()
		
		# complete processes
		for p in processes:
			p.join()
	
	# listen
	flag = [0]*nprocessor
	over = False
	while over==False:
		time.sleep(1)
		over = True
		for ino in range(0,nprocessor):
			mono_file = Change_ext(t_mzXML_fullfile,'_Mono%d.npz' % (ino+1))
			if False==os.path.isfile(mono_file):
				if flag[ino]==0:
					flag[ino] = 1
					#print('Deisotoping: batch%d' % (ino+1))
				over = False
				#break
	#print('\nFinish deisotoping: %s' % tname)
	
	return
	
def output_ms2scan(file1,nout_format,fname,MS2_index,MS2_peaks,index,ms2scan,mononum,monomz,monochg,monointen,monoppi,cno,cur_nbatch,ncali):
	# output each ms2scan
	
	# MS2_index=zeros((MaxMSNum,8)) # MS1 scan,MS1 rt,MS2 scan,m/z,z,Fragtype,MS2 peak num,massshift
	cur_ms2scan=int(ms2scan[cno])
	sno=nonzero(MS2_index[:,2]==cur_ms2scan)[0][0]
	cur_ms1rt=MS2_index[sno,1]
	cur_premz=MS2_index[sno,3]
	cur_prech=int(MS2_index[sno,4])
	cur_massshift=MS2_index[sno,7]
	mz=MS2_peaks[index[sno]-1:index[sno+1]-1,0]
	inten=MS2_peaks[index[sno]-1:index[sno+1]-1,1]
	if ncali==1:
		precMZCorr = massCorrectionFunction([cur_premz],cur_massshift)
		cur_premz=precMZCorr[0]
		mz = massCorrectionFunction(mz,cur_massshift)
	
	cur_mononum=int(mononum[cno])
	if cur_mononum>0:
		cur_monomz=monomz[cno,0:cur_mononum]
		cur_monochg=monochg[cno,0:cur_mononum]
		cur_monointen=monointen[cno,0:cur_mononum]
		cur_monoppi=monoppi[cno,0:cur_mononum]
		if ncali==1:
			cur_monomz = massCorrectionFunction(cur_monomz,cur_massshift)
	if nout_format==1:#ms2
		if cno==0 and cur_nbatch==1:
			file1.write('H	CreationDate	%s\n' % (time.asctime( time.localtime(time.time()) )))
			file1.write('H	Extractor	DeisotopeMS1\n')
			file1.write('H	ExtractorVersion	0.0.1\n')
			file1.write('H	Comments	Owned by JUMP, 2021\n')
			file1.write('H	DataType	Centroided\n')
		if cur_mononum==0:
			cur_monomz=[]
			cur_monochg=[]
			cur_monointen=[]
			cur_monoppi=[]
			# cur_monomz.append(cur_premz)
			# cur_monochg.append(1)
			# cur_monointen.append(0)
			# cur_monoppi.append(0)
			cur_monomz.append(cur_premz)
			cur_monochg.append(2)
			cur_monointen.append(0)
			cur_monoppi.append(0)
			cur_monomz.append(cur_premz)
			cur_monochg.append(3)
			cur_monointen.append(0)
			cur_monoppi.append(0)
			cur_mononum=2
		for ino in range(cur_mononum):
			charge=cur_monochg[ino]
			premz=cur_monomz[ino]
			pre_monointen=cur_monointen[ino]
			pre_monoppi=cur_monoppi[ino]
			if charge==0:
				write_ms2(file1,cur_ms2scan,2,cur_ms1rt,premz,pre_monointen,pre_monoppi,mz,inten)
				write_ms2(file1,cur_ms2scan,3,cur_ms1rt,premz,pre_monointen,pre_monoppi,mz,inten)
			else:
				write_ms2(file1,cur_ms2scan,charge,cur_ms1rt,premz,pre_monointen,pre_monoppi,mz,inten)
	elif nout_format==2:#dtas
		if cur_mononum==0:
			# charge=1
			# title='%s.%d.1.%d.dta' % (fname,cur_ms2scan,charge)
			# write_dtas(file1,title,charge,cur_premz,mz,inten)
			charge=2
			title='%s.%d.1.%d.dta' % (fname,cur_ms2scan,charge)
			write_dtas(file1,title,charge,cur_premz,mz,inten)
			charge=3
			title='%s.%d.1.%d.dta' % (fname,cur_ms2scan,charge)
			write_dtas(file1,title,charge,cur_premz,mz,inten)
		else:
			for ino in range(cur_mononum):
				charge=cur_monochg[ino]
				title='%s.%d.%d.%d.dta' % (fname,cur_ms2scan,ino+1,charge)
				premz=cur_monomz[ino]
				if charge==0:
					charge=2
					title='%s.%d.%d.%d.dta' % (fname,cur_ms2scan,ino+1,charge)
					write_dtas(file1,title,charge,premz,mz,inten)
					charge=3
					title='%s.%d.%d.%d.dta' % (fname,cur_ms2scan,ino+1,charge)
					write_dtas(file1,title,charge,premz,mz,inten)
				else:
					write_dtas(file1,title,charge,premz,mz,inten)
	else:#mgf
		if cur_mononum==0:
			# charge=1
			# title='%s.%d.1.%d.dta' % (fname,cur_ms2scan,charge)
			# write_mgf(file1,title,charge,cur_ms1rt,cur_premz,mz,inten)
			charge=2
			title='%s.%d.1.%d.dta' % (fname,cur_ms2scan,charge)
			write_mgf(file1,title,charge,cur_ms1rt,cur_premz,mz,inten)
			charge=3
			title='%s.%d.1.%d.dta' % (fname,cur_ms2scan,charge)
			write_mgf(file1,title,charge,cur_ms1rt,cur_premz,mz,inten)
		else:
			for ino in range(cur_mononum):
				charge=cur_monochg[ino]
				title='%s.%d.%d.%d.dta' % (fname,cur_ms2scan,ino+1,charge)
				premz=cur_monomz[ino]
				if charge==0:
					charge=2
					title='%s.%d.%d.%d.dta' % (fname,cur_ms2scan,ino+1,charge)
					write_mgf(file1,title,charge,cur_ms1rt,premz,mz,inten)
					charge=3
					title='%s.%d.%d.%d.dta' % (fname,cur_ms2scan,ino+1,charge)
					write_mgf(file1,title,charge,cur_ms1rt,premz,mz,inten)
				else:
					write_mgf(file1,title,charge,cur_ms1rt,premz,mz,inten)
	return

def OutputMono(t_mzXML_fullfile,jump_params,cur_nbatch,chg_nums):
	# output mono peaks
	
	# get internal parameters
	params = Get_params(jump_params)
	nw_wo_cali=params['w_wo_calibration']
	nout_format=params['output_format']
	if nout_format==1:
		c_format='.ms2'
	elif nout_format==2:
		c_format='.dtas'
	else:
		c_format='.mgf'
	
	# load ms1 and ms2
	[resultpath,tname] = os.path.split(t_mzXML_fullfile)
	[fname,ext] = os.path.splitext(tname)
	
	if nw_wo_cali==2:
		MS2_outfile1 = Change_ext(t_mzXML_fullfile,'.raw'+c_format)
	else:
		MS2_outfile1 = Change_ext(t_mzXML_fullfile,c_format)
	if nw_wo_cali!=1 and nw_wo_cali!=2:
		MS2_outfile2 = Change_ext(t_mzXML_fullfile,'.raw'+c_format)
	# if os.path.isfile(MS2_outfile1):
		# return
	
	MS2_scanfile = Change_ext(t_mzXML_fullfile,'_MS2scans.npz')
	MS2_peakfile = Change_ext(t_mzXML_fullfile,'_MS2peaks.npz')
	MS2_monofile = Change_ext(t_mzXML_fullfile,'_Mono%d.npz' % (cur_nbatch))
	
	M2scan=load(MS2_scanfile)
	M2peak=load(MS2_peakfile)
	MS2_index=M2scan['MS2_index']
	MS2_peaks=M2peak['MS2_peaks']
	num_MS2=MS2_index.shape[0]
	M2mono=load(MS2_monofile)
	ms2scan=M2mono['ms2scan']
	mononum=M2mono['mononum']
	monomz=M2mono['monomz']
	monochg=M2mono['monochg']
	monointen=M2mono['monointen']
	monoppi=M2mono['monoppi']
	cur_num_MS2=monomz.shape[0]
	#print([num_MS1,num_MS2])
	index0=[1]+list(MS2_index[0:num_MS2,6])
	index=[int(i) for i in index0]
	#print([index[0],index[1],len(index)])
	
	file1 = open(MS2_outfile1,'a')
	if nw_wo_cali!=1 and nw_wo_cali!=2:
		file2 = open(MS2_outfile2,'a')
	for cno in range(0,cur_num_MS2):
		if nw_wo_cali==1:
			output_ms2scan(file1,nout_format,fname,MS2_index,MS2_peaks,index,ms2scan,mononum,monomz,monochg,monointen,monoppi,cno,cur_nbatch,1)
		elif nw_wo_cali==2:
			output_ms2scan(file1,nout_format,fname,MS2_index,MS2_peaks,index,ms2scan,mononum,monomz,monochg,monointen,monoppi,cno,cur_nbatch,0)
		else:
			output_ms2scan(file1,nout_format,fname,MS2_index,MS2_peaks,index,ms2scan,mononum,monomz,monochg,monointen,monoppi,cno,cur_nbatch,1)
			output_ms2scan(file2,nout_format,fname,MS2_index,MS2_peaks,index,ms2scan,mononum,monomz,monochg,monointen,monoppi,cno,cur_nbatch,0)
		
	file1.close()
	if nw_wo_cali!=1 and nw_wo_cali!=2:
		file2.close()
	
	# collect charges
	for cno in range(0,cur_num_MS2):
		cur_mononum=int(mononum[cno])
		if cur_mononum>0:
			cur_monochg=monochg[cno,0:cur_mononum]
			cur_monoppi=monoppi[cno,0:cur_mononum]
			cur_monochg = cur_monochg.tolist()
			cur_monoppi = cur_monoppi.tolist()
			idx = cur_monoppi.index(max(cur_monoppi))
			s_chg = int(cur_monochg[idx])
			if s_chg<=5:
				chg_nums[s_chg]=chg_nums[s_chg]+1
		else:
			chg_nums[0]=chg_nums[0]+1
	
	return chg_nums

def remove_fullfiles(fullfiles):
	# remove files
	if len(fullfiles)>0:
		for f in fullfiles:
			os.remove(f)

def remove_temp_files(t_mzXML_fullfile):
	# remove batch files
	[resultpath,tname] = os.path.split(t_mzXML_fullfile)
	[npz_fullfiles,npz_files] = OneFormat(resultpath,'*.npz')
	remove_fullfiles(npz_fullfiles)
	[params_fullfiles,params_files] = OneFormat(resultpath,'*.deiso.params')
	remove_fullfiles(params_fullfiles)
	[sh_fullfiles,sh_files] = OneFormat(resultpath,'*.deiso.sh')
	remove_fullfiles(sh_fullfiles)
	[bat_fullfiles,bat_files] = OneFormat(resultpath,'*.deiso.bat')
	remove_fullfiles(bat_fullfiles)
	
	return

def PostProcessDeisotope(mzXML_path,sample):
	
	dtas = glob.glob(mzXML_path+"/"+sample+"/"+sample+".*.dtas")
	if len(dtas)==0:
		value = 0
	else:
		suffixes = []
		for dta in dtas:
			suffix = float(re.search(sample+".(\d+).dtas",os.path.split(dta)[1]).group(1))
			suffixes.append(int(suffix))
		value = max(suffixes)
	
	# create dir
	mzXML_sample_path = mzXML_path+"/"+sample
	if False==os.path.isdir(mzXML_sample_path):
		os.mkdir(mzXML_sample_path)
	mzXML_samplex2_path = mzXML_path+"/"+sample+"/"+sample+"."+str(value+1)
	if False==os.path.isdir(mzXML_samplex2_path):
		os.mkdir(mzXML_samplex2_path)
	
	# move files
	if True==os.path.isfile(os.path.join(mzXML_sample_path,sample+".dtas")):
		shutil.move(os.path.join(mzXML_sample_path,sample+".dtas"),mzXML_samplex2_path+".dtas")

def Rel2Abs(filename,ncase):
	
	[tmppath,tmpname] = os.path.split(filename)
	[fname,ext] = os.path.splitext(tmpname)
	sample_filename = os.path.join(tmppath,fname,tmpname)
	if len(tmppath)==0 or tmppath=='.':
		filename_abs1 = os.path.join(os.getcwd(),tmpname)
		filename = filename_abs1
		filename_abs2 = os.path.join(os.getcwd(),fname,tmpname)
		sample_filename = filename_abs2
	
	if ( ncase==1 and False==os.path.isfile(filename) ) or ( ncase!=1 and False==os.path.isfile(filename) and False==os.path.isfile(sample_filename) ):
		print('file does not exist: '+filename)
		filename=''
	
	return filename

def msg(name=None):
	return '''\n\npython DeisotopeMS1.py jump_preprocess_paramfile file.mzXML\n\n'''



def main():
	# ******start****** #
	start_time = time.time()
	
	# args
	parser = argparse.ArgumentParser(description="JUMP Preprocessing", prog="V2.3.0",usage=msg())
	parser.add_argument("jump_preprocess_paramfile", help="jump preprocess parameter file")
	parser.add_argument("mzXML",help="single or list of mzXML/mzML files",nargs='+')
	args = parser.parse_args()
	
	# jump_params
	jump_params = args.jump_preprocess_paramfile
	jump_params = Rel2Abs(jump_params,1)
	if len(jump_params)==0:
		return
	
	# mzXMLs
	mzXMLs = args.mzXML
	mzXML_path = os.getcwd()
	# ++++ raw_data_type ++++
	raw_data_type = "mzXML"
	if '.mzML' in mzXMLs[0]:
		raw_data_type = "mzML"
	
	if mzXMLs == ["*.{}".format(raw_data_type)]:# '*.mzXML'
		mzXMLs = glob.glob(mzXML_path+"/*.{}".format(raw_data_type))
		if len(mzXMLs)==0:
			print('{} does not exist: {}'.format(raw_data_type,mzXML_path))
			return
	elif len(mzXMLs)==1 and "*.{}".format(raw_data_type) in mzXMLs[0]:# 'datapath/*.mzXML'
		datapath=os.path.split(mzXMLs[0])[0]
		mzXMLs = glob.glob(mzXMLs[0])
		if len(mzXMLs)==0:
			print('{} does not exist: {}'.format(raw_data_type,datapath))
			return
	else:
		mzXMLs_new =[]
		for mz_F in mzXMLs:
			mz_F = Rel2Abs(mz_F,2)
			if len(mz_F)==0:
				return
			mzXMLs_new.append(mz_F)
		mzXMLs = mzXMLs_new
	# mzXML_fullfiles
	mzXML_fullfiles = mzXMLs
	mzXML_fullfiles.sort()
	
	# codepath
	cur_path = os.getcwd()
	[codepath,tmpname] = os.path.split(sys.argv[0].strip())
	if len(codepath)==0 or codepath=='.':
		codepath = os.getcwd()
	if False==os.path.isdir(codepath):
		print('codepath does not exist: '+codepath)
		return
	os.chdir(codepath)
	
	warnings.filterwarnings("ignore")
	
	# params
	params = Get_params(jump_params)
	tmt=params['tmt']
	nrunning_mode=params['deisotoping_method']
	nprocessor=params['nprocessor']
	nout_format=params['output_format']
	if nout_format==1:
		c_format='.ms2'
	elif nout_format==2:
		c_format='.dtas'
	else:
		c_format='.mgf'
	
	print("\n################################################################")
	print("#       ****  Using JUMP pre-processing             ****       #")
	print("################################################################\n")
	
	print("[param:%s" % (jump_params))
	print("  Using the following rawfiles:")
	for c_mzXML_fullfile in mzXML_fullfiles:
		mz_F=os.path.split(c_mzXML_fullfile)[1]
		print("  %s" % (mz_F))
	print("")
	
	# process mzXML
	for c_mzXML_fullfile in mzXML_fullfiles:
		[tpath,tname] = os.path.split(c_mzXML_fullfile)
		[fname,ext] = os.path.splitext(tname)
		resultpath = os.path.join(tpath,fname)
		Check_Path(resultpath)
		t_mzXML_fullfile = os.path.join(resultpath,tname)
		
		if True==os.path.isfile(c_mzXML_fullfile):
			shutil.move(c_mzXML_fullfile,resultpath)
		
		now = datetime.now()
		dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
		print("  Preprocessing data: %s" % (tname))
		print("  Start: %s\n" % (dt_string))
		
		print("  Converting .raw into .{} file".format(raw_data_type))
		print("  {} file (./{}) has already existed".format(raw_data_type,tname))
		print("  Extracting peaks from .{}".format(raw_data_type))
		[nMS1,nMS2,scan_ppm]=ReadmzXML(t_mzXML_fullfile,raw_data_type,tmt,nrunning_mode)
		print("  Gathering scan information: %d of %d scans" % (nMS1+nMS2,nMS1+nMS2))
		print("  There are %d MS and %d MS/MS in the entire run\n" % (nMS1,nMS2))
		
		print("  Mass correction")
		if len(scan_ppm)==0:
			if len(tmt)==0:
				print("  Mass correction will not be performed when less than 5%% of spectra is used")
			else:
				print("  Any of reference ions (TMTreporter-126, y1-ions of K and R) is not found")
			print("  No mass-shift correction\n")
		else:
			print("  Use %d bin to correct the mass shift" % (len(scan_ppm)))
			for ino in range(1,len(scan_ppm)+1):
				print("  Mass-shift at bin %d: mean = %.5f ppm" % (ino,scan_ppm[ino-1][1]))
			print("  correcting MS1 scan")
			print("  correcting MS2 scan\n")
			
			print("  Mass-shift correction has been finished\n")
		
		GetMonoInBatch(t_mzXML_fullfile,raw_data_type,jump_params,scan_ppm,nrunning_mode)
		
		if nrunning_mode!=4:# non-debug mode
			MS2_outfile1 = Change_ext(t_mzXML_fullfile,'.raw'+c_format)
			MS2_outfile2 = Change_ext(t_mzXML_fullfile,c_format)
			if True==os.path.isfile(MS2_outfile1):
				os.remove(MS2_outfile1)
			if True==os.path.isfile(MS2_outfile2):
				os.remove(MS2_outfile2)
			
			chg_nums=array([0]*6)
			for ino in range(0,nprocessor):
				chg_nums=OutputMono(t_mzXML_fullfile,jump_params,ino+1,chg_nums)
			
			remove_temp_files(t_mzXML_fullfile)
			PostProcessDeisotope(tpath,fname)
			
			now = datetime.now()
			dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
			print("\n\n  Decharging scans")
			print("  Decharged %d precursor ions: 0=%d 1=%d 2=%d 3=%d 4=%d 5=%d" % (sum(chg_nums),chg_nums[0],chg_nums[1],chg_nums[2],chg_nums[3],chg_nums[4],chg_nums[5]))
			print("  PPI distribution: 1 = %d" % (sum(chg_nums)+chg_nums[0]))
			print("  Date: %s\n" % (dt_string))
	
	# ******end****** #
	os.chdir(cur_path)
	end_time = time.time()
	total_time = end_time-start_time
	print('\n  Total preprocessing time: %.1f min\n' % (total_time/60))
	return

if __name__ == '__main__':
	main()
