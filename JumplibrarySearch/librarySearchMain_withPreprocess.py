import os, sys, glob
import argparse
from logFunctions import *
import configparser
import subprocess
import re
import utils
import time
config = configparser.ConfigParser()

from os.path import dirname


def msg(name=None):
    return '''\n\n\npython mainProgram.py params_file\n\n\n '''

parser = argparse.ArgumentParser(description="JUMP spectral library search", prog="mainProgramServer.py",usage=msg())
parser.add_argument("parameterfile", help="Jump-lib-s parameter file")
parser.add_argument("--queue",dest="queue", default="queue=standard")
parser.add_argument("--mem",dest="mem", default="mem=20G")

args = parser.parse_args()
queue = args.queue.split("=")[-1]
mem = args.mem.split("=")[-1]

params_file = args.parameterfile


# path for the script
source_path_script = sys.argv[0]
#get the path of script
source_path = dirname(source_path_script)

preprocess_param_file = "{}/preprocess/v2.3/parameterFile/jump_preprocess.params".format(dirname(source_path))

# print (preprocess_param_file)

config.read(params_file)

mzxml_path = config["specLib"]["mzxml_path"]
# ms2_path = config["specLib"]["ms2_path"]
job_type = config["specLib"]["job_type"]

# ++++ raw_data_type ++++
raw_data_type = "mzXML"
if len( glob.glob(mzxml_path+"/*.mzML") )>0:
    raw_data_type = "mzML"

# ms2_files_all = glob.glob(ms2_path+"/*.ms2")
mz_files = glob.glob(mzxml_path+"/*.{}".format(raw_data_type))
# print (mz_files)

def create_job_file(params_file,ms2File):
    fileroot = ms2File.split("/")[-1].split(".{}".format(raw_data_type))[0]
    makedirectory(fileroot)
    path = os.getcwd()

    log_dir = path+"/"+fileroot+"/log"
    os.system("mkdir "+log_dir)
    job_header = "#!/bin/bash\n#BSUB -P JUMP-lib-s\n#BSUB -J spectral library search\n#BSUB -oo "+log_dir+"/log.out\n#BSUB -eo "+log_dir+"/error.err\n#BSUB -n 1\n"
    job_body_pre = "python {}/preprocess/v2.3/DeisotopeMS1.py {} {}\n".format(dirname(source_path),"jump_preprocess.params",ms2File) 
    ms2File = ms2File.split(".{}".format(raw_data_type))[0]+"/"+fileroot+".ms2"
    job_body1 = "python {}/librarySearch.py {} {}".format(source_path,params_file,ms2File)
    # print (job_body1)
    jobfile = fileroot+".sh"
    with open(jobfile,"w") as new_file:
        new_file.write(job_header+job_body_pre+job_body1)
    return jobfile



def checkJobStatus(jobNumbers):
    nRemainder = len(jobNumbers)
    while nRemainder > 0:
        time.sleep(10)
        p = subprocess.Popen("bjobs -noheader", shell=True, stdout=subprocess.PIPE)
        (pout, _) = p.communicate()
        pout = pout.decode().strip().split("\n")
        jobRemainder = [p.split()[0] for p in pout]
        nRemainder = len(set(jobNumbers).intersection(jobRemainder))
        nFinished = len(jobNumbers) - nRemainder
        text = "\r  {} job(s) is/are finished".format(nFinished)
        sys.stdout.write(text)
        sys.stdout.flush()
    print ("  {} job(s) is/are finished".format(nFinished))

def submit_job(jobf,queue,mem):
    cmd = 'bsub -q '+queue+' -R "rusage[mem='+mem+']" < '+jobf
    # os.system(cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    (pout, _) = p.communicate()
    pout = pout.decode().strip()
    pout = re.search("(?<=Job <)([0-9]+)", pout)
    jobNumber = pout.group(0)
    return jobNumber

cp_pp_params = "cp {} .".format(preprocess_param_file)
os.system(cp_pp_params)

jobNumbers = []
for mzFile in mz_files:
    
    if job_type == "0":   
        cmd = "python {}/preprocess/v2.3/DeisotopeMS1.py {} {}".format(dirname(source_path),"jump_preprocess.params",mzFile)
        print (cmd)
        # cmd = "python /home/spoudel1/bin/python/JUMPp-lib_v5/preprocess/v2.3/DeisotopeMS1.py "+preprocess_param_file+" "+mzFile
        os.system(cmd)
        newFolder = mzFile.split(".{}".format(raw_data_type))[0].split("/")[-1]
        ms2File = mzFile.split(".{}".format(raw_data_type))[0]+"/"+newFolder+".ms2"
        cmd =  "python {}/librarySearch.py {} {}".format(source_path,params_file,ms2File)
        print (cmd)
        os.system(cmd)
    else:
        job_num=submit_job(create_job_file(params_file,mzFile),queue,mem)
        print ("\n\nJob is submitted for spectral library searching for {}\n".format(mzFile))
        jobNumbers.append(job_num)

if job_type != "0":
    checkJobStatus(jobNumbers)




