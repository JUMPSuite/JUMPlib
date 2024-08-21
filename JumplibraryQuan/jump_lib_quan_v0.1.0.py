import os, sys, re, numpy as np, pandas as pd, scipy.stats as stats,matplotlib.pyplot as plt
from pyteomics import ms2, mzxml, mzml
from scipy.stats import t
import numpy.ma as ma
from collections import OrderedDict, Counter
from scipy.stats import spearmanr, pearsonr
from process_ms3 import *

class OrderedCounter(Counter, OrderedDict):
	pass

###############################################################
# small modules/functions                 
###############################################################                                                       

def getParams(paramFile):
    parameters = dict()
    with open(paramFile, 'r') as file:
        for line in file:
            if re.search(r'^#', line) or re.search(r'^\s', line):
                continue
            line = re.sub(r'#.*', '', line)  # Remove comments (start from '#')
            line = re.sub(r'\s*', '', line)  # Remove all whitespaces

            # Exception for "feature_files" parameter
            if "feature_files" in parameters and line.endswith("feature"):
                parameters["feature_files"].append(line)
            else:
                key = line.split('=')[0]
                val = line.split('=')[1]
                if key == "feature_files":
                    parameters[key] = [val]
                else:
                    parameters[key] = val
    return parameters


class progressBar:
    def __init__(self, total):
        self.total = total
        self.barLength = 20
        self.count = 0
        self.progress = 0
        self.block = 0
        self.status = ""

    def increment(self):
        self.count += 1
        self.progress = self.count / self.total
        self.block = int(round(self.barLength * self.progress))
        if self.progress == 1:
            self.status = "Done...\r\n"
        else:
            self.status = ""

        text = "\r  Progress: [{0}] {1}% {2}".format("#" * self.block + "-" * (self.barLength - self.block), int(self.progress * 100), self.status)
        sys.stdout.write(text)
        sys.stdout.flush()


def getReporterMz(name):
    if name == "sig126":
        return 126.127725938
    elif name == "sig127" or name == "sig127N":
        return 127.124760831
    elif name == "sig127C":
        return 127.131080776
    elif name == "sig128N":
        return 128.128115669
    elif name == "sig128" or name == "sig128C":
        return 128.134435614
    elif name == "sig129" or name == "sig129N":
        return 129.131470507
    elif name == "sig129C":
        return 129.137790451
    elif name == "sig130N":
        return 130.134825345
    elif name == "sig130" or name == "sig130C":
        return 130.141145289
    elif name == "sig131" or name == "sig131N":
        return 131.138180183
    elif name == "sig131C":
        return 131.144500127
    elif name == "sig132N":
        return 132.141535020
    elif name == "sig132C":
        return 132.147854965
    elif name == "sig133N":
        return 133.144889858
    elif name == "sig133C":
        return 133.151209803
    elif name == "sig134N":
        return 134.148244696
    elif name == "sig134C":
        return 134.154565
    elif name == "sig135N":
        return 135.1515995

def getReporterIntensity(spec, params, **kwargs):
    tol = 10
    reporterNames = params["tmt_reporters_used"].split(";")
    mzArray = []
    intensityArray = []
    
    for reporter in reporterNames:
        if reporter in kwargs:
            mz = getReporterMz(reporter) * (1 + kwargs[reporter]['meanMzShift'] / 1e6)
            tol = kwargs[reporter]['sdMzShift'] * np.float(params['tmt_peak_extraction_second_sd'])
        else:
            mz = getReporterMz(reporter)

        lL = mz - mz * tol / 1e6
        uL = mz + mz * tol / 1e6
        ind = np.where((spec["m/z array"] >= lL) & (spec["m/z array"] <= uL))[0]
        if len(ind) == 0:
            reporterMz = 0
        elif len(ind) == 1:
            ind = ind[0]
            reporterMz = spec["m/z array"][ind]
        elif len(ind) > 1:
            if params['tmt_peak_extraction_method'] == '2':
                ind2 = np.argmin(abs(mz - spec["m/z array"][ind]))
                ind = ind[ind2]
                reporterMz = spec["m/z array"][ind]
            else:
                ind2 = np.argmax(spec["intensity array"][ind])
                ind = ind[ind2]
                reporterMz = spec["m/z array"][ind]
        if lL <= reporterMz < uL:
            reporterIntensity = spec["intensity array"][ind]
        else:
            reporterIntensity = 0
        mzArray.append(reporterMz)
        intensityArray.append(reporterIntensity)
        
    outArray = mzArray + intensityArray
    return outArray

def getReporterSummary(df, reporters):
    print("  Summary of quantified TMT reporter ions")
    res = {}
    for reporter in reporters:
        res[reporter] = {}
        reporterMz = getReporterMz(reporter)
        measuredMz = df[reporter.replace("sig", "mz")]
        measuredMz = measuredMz[measuredMz > 0]
        n = len(measuredMz)
        meanMzShift = ((measuredMz - reporterMz) / reporterMz * 1e6).mean()
        sdMzShift = ((measuredMz - reporterMz) / reporterMz * 1e6).std(ddof=1)
        res[reporter]['nPSMs'] = n
        res[reporter]['meanMzShift'] = meanMzShift
        res[reporter]['sdMzShift'] = sdMzShift

    return res

def readerToNumpyArr(reader,raw_data_type):
    #mzXcols = ["num","msLevel","m/z array","intensity array"]
    if raw_data_type=="mzXML":
        data = []
        
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
            
            if msLevel==3 or (msLevel==2 and ms3level==1 and num not in ms123scan_ms2_array):
                continue
            
            mz_array = spec["m/z array"]
            intensity_array = spec["intensity array"]
            
            if msLevel==2 and ms3level==1 and num in ms123scan_ms2_array:
                pos1=np.nonzero( ms123scan_ms2_array==num )[0]
                if len(pos1)>0:
                    ms3scan = ms123scan_ms3_list[pos1[0]]
                    ms3_spec = reader[ms3scan-1]
                    ms3mz = ms3_spec["m/z array"]
                    ms3inten = ms3_spec["intensity array"]
                    [mz_array,intensity_array] = Merge_ms23(mz_array,intensity_array,ms3mz,ms3inten)
            
            spectrum_data = {
                "num": num,
                "msLevel": msLevel,
                "m/z array": mz_array,
                "intensity array": intensity_array
            }
            
            data.append(spectrum_data)
        
        dfmzXML = pd.DataFrame(data)
    elif raw_data_type=="mzML":
        data = []
        
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
            
            if msLevel==3 or (msLevel==2 and ms3level==1 and num not in ms123scan_ms2_array):
                continue
            
            mz_array = spec["m/z array"]
            intensity_array = spec["intensity array"]
            
            if msLevel==2 and ms3level==1 and num in ms123scan_ms2_array:
                pos1=np.nonzero( ms123scan_ms2_array==num )[0]
                if len(pos1)>0:
                    ms3scan = ms123scan_ms3_list[pos1[0]]
                    ms3_spec = reader[ms3scan-1]
                    ms3mz = ms3_spec["m/z array"]
                    ms3inten = ms3_spec["intensity array"]
                    [mz_array,intensity_array] = Merge_ms23(mz_array,intensity_array,ms3mz,ms3inten)
            
            spectrum_data = {
                "num": num,
                "msLevel": msLevel,
                "m/z array": mz_array,
                "intensity array": intensity_array
            }
            
            data.append(spectrum_data)
        
        dfmzXML = pd.DataFrame(data)
    elif raw_data_type=="ms2":
        data = []
        
        for ino in range(0,len(reader)):
            spec = reader[ino]
            num = int(spec["params"]["scan"][0])
            msLevel = 2
            
            mz_array = spec["m/z array"]
            intensity_array = spec["intensity array"]
            
            spectrum_data = {
                "num": num,
                "msLevel": msLevel,
                "m/z array": mz_array,
                "intensity array": intensity_array
            }
            
            data.append(spectrum_data)
        
        dfmzXML = pd.DataFrame(data)
    else:
        data = {"num": [], "msLevel": [], "m/z array": [], "intensity array": []}
        dfmzXML = pd.DataFrame(data)
    
    return dfmzXML

def extractReporters(files, df, params, **kwargs):
    # Input arguments
    # files: mzXML or ms2 files to be quantified
    # df: dataframe of ID.txt file
    # params: parameters

    if "sig126" in kwargs:
        print("\n  Refined extraction of TMT reporter ion peaks")
    else:
        print("\n  Extraction of TMT reporter ion peaks")

    dictQuan = {}
    for file in files:
        print("    Working on {}".format(os.path.basename(file)))
        ext = os.path.splitext(file)[-1]
        # if ext == ".mzXML":
            # reader = mzxml.MzXML(file)  # mzXML file reader
        # elif ext == ".ms2":
            # reader = ms2.IndexedMS2(file)  # MS2 file reader
        # else:
            # sys.exit(" Currently, either .mzXML or .ms2 file is supported")
        
        if ext==".mzXML" or ext==".mzML" or ext==".ms2":
            if ext==".mzXML":
                reader = mzxml.MzXML(file)
            elif ext==".mzML":
                reader = mzml.MzML(file)
            elif ext==".ms2":
                reader = ms2.IndexedMS2(file)
            
            raw_data_type = ext[1:]
            
            dfmzinten = readerToNumpyArr(reader,raw_data_type)
        else:
            data = {"num": [], "msLevel": [], "m/z array": [], "intensity array": []}
            dfmzinten = pd.DataFrame(data)
            sys.exit(" Currently, either .mzXML, .mzML, or .ms2 file is supported")

        # Extraction of TMT reporter ions in each fraction
        scans = list(df['scan'][df['frac'] == file].unique())
        progress = progressBar(len(scans))
        for scan in scans:
            progress.increment()
            # spec = reader[str(scan)]
            # spec = reader[int(scan)-1]
            filtered_df = dfmzinten.loc[dfmzinten.num==int(scan)]
            for _, spec in filtered_df.iterrows():
                break
            res = getReporterIntensity(spec, params, **kwargs)  # Array of reporter m/z and intensity values
            key = file + "_" + str(scan)
            dictQuan[key] = res

    # Create a dataframe of quantification data
    reporters = params["tmt_reporters_used"].split(";")
    colNames = [re.sub("sig", "mz", i) for i in reporters] + reporters
    res = pd.DataFrame.from_dict(dictQuan, orient='index', columns=colNames)

    # Summary of quantified TMT reporter ions
    print()
    reporterSummary = getReporterSummary(res, reporters)
    nTot = len(res)
    for reporter in reporters:
        n = reporterSummary[reporter]["nPSMs"]
        print("    %s\t%d (%.2f%%) matched" % (reporter, n, n / nTot * 100))

    return res, reporterSummary


def correctImpurity(df, params):
    if params['impurity_correction'] == "1":
        reporters = params["tmt_reporters_used"].split(";")
        dfImpurity = pd.read_table(params["impurity_matrix"], sep="\t", skiprows=1, header=None, index_col=0)
        dfImpurity = pd.DataFrame(np.linalg.pinv(dfImpurity.values), dfImpurity.columns, dfImpurity.index)
        dfCorrected = df[reporters].dot(dfImpurity.T)
        dfCorrected.columns = reporters
        df[reporters] = pd.concat([df[reporters]/2, dfCorrected]).groupby(level=0).max()

    return df


def getSubset(df, params):
    # Get a subset of a dataframe to calculate loading-bias information
    # 1. Filter out PSMs based on the intensity level
    reporters = params["tmt_reporters_used"].split(";")
    noiseLevel = 1000
    snRatio = float(params["SNratio_for_correction"])
    subDf = df[reporters][(df[reporters] > noiseLevel * snRatio).prod(axis=1).astype(bool)]  # Zero-intensity PSMs are excluded

    # 2. Filter out highly variant PSMs in each column (reporter)
    psmMean = subDf.mean(axis=1)
    subDf = np.log2(subDf.divide(psmMean, axis=0))
    pctTrimmed = float(params["percentage_trimmed"])
    n = 0
    for reporter in reporters:
        if n == 0:
            ind = ((subDf[reporter] > subDf[reporter].quantile(pctTrimmed / 200)) &
                   (subDf[reporter] < subDf[reporter].quantile(1 - pctTrimmed / 200)))
        else:
            ind = ind & ((subDf[reporter] > subDf[reporter].quantile(pctTrimmed / 200)) &
                         (subDf[reporter] < subDf[reporter].quantile(1 - pctTrimmed / 200)))
        n += 1

    subDf = subDf.loc[ind]
    return subDf


def getLoadingBias(df, params):
    ###########################
    # Loading-bias evaluation #
    ###########################
    subDf = getSubset(df, params)
    n = len(subDf)
    sm = 2 ** subDf.mean(axis=0)    # Sample-mean values
    msm = np.mean(sm)    # Mean of sample-mean values
    avg = sm / msm * 100
    sdVal = subDf.std(axis=0, ddof = 1)
    sd = ((2 ** sdVal - 1) + (1 - 2 ** (-sdVal))) / 2 * 100
    sem = sd / np.sqrt(n)
    return avg, sd, sem, n


def normalization(df, params):
    ################################################
    # Normalization (i.e. loading-bias correction) #
    ################################################
    doNormalization = params["loading_bias_correction"]
    normalizationMethod = params["loading_bias_correction_method"]
    if doNormalization == "1":
        # First, get a subset for calculating normalization factors (same as loading-bias calculation)
        # Note that this subset is 1) divided by row-wise mean (i.e. PSM-wise mean) and then 2) log2-transformed
        subDf= getSubset(df, params)
        # Calculate normalization factors for samples (reporters)
        print("  Normalization is being performed")
        if normalizationMethod == "1":  # Trimmed-mean
            sm = subDf.mean(axis=0)
        elif normalizationMethod == "2":  # Trimmed-median
            sm = subDf.median(axis=0)
        target = np.mean(sm)
        normFactors = sm - target
        # Normalize the input dataframe, df (in log2-scale and then scale-back)
        res = df.copy()
        psmMeans = res[reporters].mean(axis=1)
        res[reporters] = np.log2(res[reporters].divide(psmMeans, axis=0).replace(0, np.nan))
        res[reporters] = res[reporters] - normFactors
        res[reporters] = 2 ** res[reporters]
        res[reporters] = res[reporters].multiply(psmMeans, axis=0)
        # After the normalization, no need to show loading-bias again (should be 100% for all samples)
    else:
        print("  Normalization is skipped according to the parameter")

    return res

def Qtest(data, left=True, right=True, alpha=0.05):
    """
    From https://sebastianraschka.com/Articles/2014_dixon_test.html#implementing-a-dixon-q-test-function
    Keyword arguments:
        data = A ordered or unordered list of data points (int or float).
        left = Q-test of minimum value in the ordered list if True.
        right = Q-test of maximum value in the ordered list if True.
        q_dict = A dictionary of Q-values for a given confidence level,
            where the dict. keys are sample sizes N, and the associated values
            are the corresponding critical Q values. E.g.,
            {3: 0.97, 4: 0.829, 5: 0.71, 6: 0.625, ...}
    Returns a list of 2 values for the outliers, or None.
    E.g.,
       for [1,1,1] -> [None, None]
       for [5,1,1] -> [None, 5]
       for [5,1,5] -> [1, None]

    """

    q90 = [0.941, 0.765, 0.642, 0.56, 0.507, 0.468, 0.437,
           0.412, 0.392, 0.376, 0.361, 0.349, 0.338, 0.329,
           0.32, 0.313, 0.306, 0.3, 0.295, 0.29, 0.285, 0.281,
           0.277, 0.273, 0.269, 0.266, 0.263, 0.26
           ]
    Q90 = {n: q for n, q in zip(range(3, len(q90) + 1), q90)}
    q95 = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466,
           0.444, 0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356,
           0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312,
           0.308, 0.305, 0.301, 0.29
           ]
    Q95 = {n: q for n, q in zip(range(3, len(q95) + 1), q95)}
    q99 = [0.994, 0.926, 0.821, 0.74, 0.68, 0.634, 0.598, 0.568,
           0.542, 0.522, 0.503, 0.488, 0.475, 0.463, 0.452, 0.442,
           0.433, 0.425, 0.418, 0.411, 0.404, 0.399, 0.393, 0.388,
           0.384, 0.38, 0.376, 0.372
           ]
    Q99 = {n: q for n, q in zip(range(3, len(q99) + 1), q99)}

    if isinstance(data, list):
        pass
    else:
        x = list(data)

    if alpha == 0.1:
        q_dict = Q90
    elif alpha == 0.05:
        q_dict = Q95
    elif alpha == 0.01:
        q_dict = Q99

    assert(left or right), 'At least one of the variables, `left` or `right`, must be True.'
    assert(len(data) >= 3), 'At least 3 data points are required'
    assert(len(data) <= max(q_dict.keys())), 'Sample size too large'

    sdata = sorted(data)
    Q_mindiff, Q_maxdiff = (0,0), (0,0)

    if left:
        Q_min = (sdata[1] - sdata[0])
        try:
            Q_min /= (sdata[-1] - sdata[0])
        except ZeroDivisionError:
            pass
        Q_mindiff = (Q_min - q_dict[len(data)], sdata[0])

    if right:
        Q_max = abs((sdata[-2] - sdata[-1]))
        try:
            Q_max /= abs((sdata[0] - sdata[-1]))
        except ZeroDivisionError:
            pass
        Q_maxdiff = (Q_max - q_dict[len(data)], sdata[-1])

    if not Q_mindiff[0] > 0 and not Q_maxdiff[0] > 0:
        outliers = []
    elif Q_mindiff[0] == Q_maxdiff[0]:
        outliers = [Q_mindiff[1], Q_maxdiff[1]]
    elif Q_mindiff[0] > Q_maxdiff[0]:
        outliers = [Q_mindiff[1]]
    else:
        outliers = [Q_maxdiff[1]]

    outlierInd = [i for i, v in enumerate(data) if v in outliers]
    # survivedInd = np.setdiff1d(range(len(data)), outlierInd)

    return outlierInd

def ESDtest(x, alpha, maxOLs):
    xm = ma.array(x)
    n = len(xm)
    R, L, minds = [], [], []
    for i in range(maxOLs):
        # Compute mean and std of x
        xmean = xm.mean()
        xstd = xm.std(ddof=1)
        # Find maximum deviation
        rr = np.abs((xm - xmean) / xstd)
        minds.append(np.argmax(rr))
        R.append(rr[minds[-1]])
        p = 1.0 - alpha / (2.0 * (n - i))
        perPoint = t.ppf(p, n - i - 2)
        L.append((n - i-1) * perPoint / np.sqrt((n - i - 2 + perPoint**2) * (n - i )))
        # Mask that value and proceed
        xm[minds[-1]] = ma.masked
    # Find the number of outliers
    ofound = False
    for i in range(maxOLs-1, -1, -1):
        if R[i] > L[i]:
            ofound = True
            break
    # Prepare return value
    if ofound:
        return minds[0:i + 1]    # There are outliers
    else:
        return []    # No outliers could be detected

def outlierRemoval(df, alpha):
    n = len(df)
    nOutliers = int(np.round(n * 0.3))
    indArray = []
    if nOutliers > n - 2:
        nOutliers = n - 2
    if nOutliers > 1:
        for i in range(df.shape[1]):
            ind = ESDtest(df.iloc[:, i], alpha, nOutliers)
            indArray.extend(ind)
    else:
        if n > 10:
            for i in range(df.shape[1]):
                ind = ESDtest(df.iloc[:, i], alpha, nOutliers)
                indArray.extend(ind)
        else:
            for i in range(df.shape[1]):
                ind = Qtest(df.iloc[:, i], alpha, nOutliers)
                indArray.extend(ind)

    # PSMs including one or more outliers will not be considered for the subsequent quantification
    indArray = list(set(indArray))    # Indices of outliers across all reporters
    df.drop(df.index[indArray], axis=0, inplace=True)
    return df

def summarization(df, inputDict, params):
    # Input arguments
    # df: a dataframe containing PSM-level quantification information
    # inputDict: a dictionary containing the relationship between protein (or peptide) and PSMs
    #            e.g., prot2Ppsm: key = each protein, value = list of PSMs corresponding to the protein
    # params: parameters from the .param file

    print("\n  Summarization of PSMs")
    resDict = {}
    reporters = params["tmt_reporters_used"].split(";")
    progress = progressBar(len(inputDict))
    for entry, psms in inputDict.items():
        progress.increment()
        psms = df.index.join(psms, how="inner").unique()
        if len(psms) == 0:
            continue
        else:
            subDf = df.loc[psms][reporters]
            # Preprocessing for outlier removal
            # 1. Log2-transformation
            # 2. PSM-wise mean calculation
            # 2.1. Representative protein abundance by the mean of top3 PSM-wise means
            #      (equivalent to the grand mean of top3 PSMs)
            # 3. Mean-centering (using the PSM-wise mean obtained at step2)
            subDf = np.log2(subDf)
            psmMeans = subDf.mean(axis=1)
            repAbundance = np.mean(sorted(psmMeans, reverse=True)[0:3])
            subDf = subDf.sub(psmMeans, axis=0)

            # Outlier removal
            if len(subDf) >= 3:
                subDf = outlierRemoval(subDf, 0.05)  # Can I make it faster?

            # Protein-level quantification (as a dictionary)
            if len(subDf) > 0:
                subDf = 2 ** (subDf.mean(axis=0) + repAbundance)
                resDict[entry] = subDf.to_dict()

    res = pd.DataFrame.from_dict(resDict, orient="index")
    return res


def getFileteredIndexes(df, method, threshold, reporters):
    # This module produces indexes to be removed (i.e., filtered indexes)
    if method == '1':  # Minimum-based filter
        idx = df[(df[reporters] < threshold).any(axis=1)].index
    elif method == '2':  # Maximum-based filter
        idx = df[(df[reporters] > threshold).any(axis=1)].index
    elif method == '3':  # Mean-based filter
        idx = df[df[reporters].mean(axis=1) < threshold].index
    elif method == '4':  # Median-based filter
        idx = df[df[reporters].median(axis=1) < threshold].index
    else:
        sys.exit("  Please check 'min_intensity_method' parameter. It should be 0, 1, 2, 3, or 4")

    return idx

def filterByIntensity(df, methods, thresholds, reporters, verbose=1):
    methodsStr = ["none", "minimum intensity", "maximum intensity", "mean intensity", "median intensity"]
    res = []
    n = 0
    for i in range(len(methods)):
        if methods[i] == "0":
            pass
        else:
            idx = getFileteredIndexes(df, methods[i], float(thresholds[i]), reporters)    # "idx" is the index to be removed
            res.extend(idx.values)
        if verbose:
            res = list(set(res))
            print("    Removed {} PSMs based on the intensity-based filter ({})".format(len(res) - n, methodsStr[int(methods[i])]))
            n = len(res)

    return res

def filterPSMs(df, prot2psm, params):
    print("\n  Examining the extracted TMT reporter ions in PSMs")
    reporters = params["tmt_reporters_used"].split(";")

    # 0. Zero-intensity filter
    n = len(df)
    df = df[(df[reporters] > 0).all(axis=1)]
    print("    Removed {} PSMs due to zero intensity at least one channel".format(n - len(df)))

    # 1. Intensity-based filtering of all PSMs
    methods = params["min_intensity_method"].split(",")
    thresholds = params["min_intensity_value"].split(",")
    idx = filterByIntensity(df, methods, thresholds, reporters, 1)  # Indexes to be removed
    idx = list(set(idx))
    df = df[~df.index.isin(idx)]

    # 2. Further filtering when only 1 or 2 PSMs are mapped to a protein
    print("    Further filtering of 1 or 2 PSMs mapped to a protein")
    methods = params["min_intensity_method_1_2_psm"].split(",")
    thresholds = params["min_intensity_value_1_2_psm"].split(",")
    idx = []
    progress = progressBar(len(prot2psm))
    for prot, psms in prot2psm.items():
        progress.increment()
        psms = df.index.join(psms, how="inner")

        if len(psms) == 0:
            continue
        elif len(psms) == 1:
            # Proteins mapped by only one PSM
            # If the PSM is filtered by the intensity-based filter, it will not be used for the quantification
            idxProt = filterByIntensity(df.loc[psms], methods, thresholds, reporters, 0)
            if len(idxProt) > 0:
                idx.extend(idxProt)
        elif len(psms) == 2:
            # Proteins mapped by two PSMs
            # Apply the intensity-based filter first
            # - If both PSMs are filtered out, they will not be used for the quantification
            # - If one of PSMs is filtered out, the PSM will not be used for the quantification
            # - If none of PSMs is filtered out, go to the next step (two PSMs can be used for the quantification)
            #   - For each PSM, check the variation (i.e., stdev) across the reporters (in log2-space)
            #   - One with smaller variation will be used for the quantification
            #   - If both PSMs have the same variation, the one with higher mean intensity will be used
            #         subDf = filterPSM1(subDf, methods, thresholds, reporters, 0)
            idxProt = filterByIntensity(df.loc[psms], methods, thresholds, reporters, 0)
            if len(idxProt) > 0:
                idx.extend(idxProt)
            else:
                psmStd = np.log2(df.loc[psms][reporters]).std(axis=1)
                psmMean = np.log2(df.loc[psms][reporters]).mean(axis=1)
                if psmStd[0] == psmStd[1]:
                    ii = np.argmin(psmMean)
                else:
                    ii = np.argmax(psmStd)
                idx.extend([psms[ii]])

    idx = list(set(idx))
    print("    Removed {} PSMs due to the larger variation than the other PSM mapped to the same protein".format(len(idx)))
    df = df[~df.index.isin(idx)]
    return df



# In[0]:
###################################################################
#                   main program                             
###################################################################
 
if __name__ == "__main__":
    ##################
    # Initialization #
    ##################
    paramFile = sys.argv[1]
    #paramFile = "jump_lib_q.params"
    params = getParams(paramFile)

    ##################
    # Parsing ID.txt #
    ##################

    # Note that this part may need to be revised according to the Jump -f result format
    print("  Loading ID.txt file")
    dfId = pd.read_table(params["idtxt"], sep="\t", skiprows=0, header=0)
    
    # ++++ raw_data_type ++++
    # path_filename_noext
    path_filename_noext = ''
    for _, row in dfId.iterrows():
        path_filename_noext = os.path.join( os.path.dirname( row['Outfile'] ), os.path.basename( row['Outfile'] ).split(".")[0] )
        break
    raw_data_type = "mzXML"
    if os.path.isfile( path_filename_noext + ".mzML" )==True:
        raw_data_type = "mzML"

    # Miscellaneous part for handling ID.txt
    #dfId["frac"] = dfId["Outfile"].apply(lambda x: os.path.dirname(x).rsplit(".", 1)[0] + ".mzXML")
    dfId["frac"] = dfId["Outfile"].apply(lambda x: os.path.dirname(x) + "/" + os.path.basename(x).split(".")[0] + "."+raw_data_type)
    dfId["scan"] = dfId["Outfile"].apply(lambda x: os.path.basename(x).split(".")[1])
    dfId["key"] = dfId["frac"] + "_" + dfId["scan"]
    fracs = dfId["frac"].unique()

    ##################################
    # Extract TMT reporter ion peaks #
    ##################################
    # 1st round of reporter ion extraction
    dfQuan, reporterSummary = extractReporters(fracs, dfId, params)

    # Before 2nd round of TMT reporter extraction, m/z-shifts of reporters are summarized
    print("\n  m/z-shift in each TMT reporter")
    reporters = params["tmt_reporters_used"].split(";")
    for reporter in reporters:
        m = reporterSummary[reporter]["meanMzShift"]
        s = reporterSummary[reporter]["sdMzShift"]
        print("    %s\tm/z-shift = %.4f [ppm]\tsd = %.4f" % (reporter, m, s))

    # 2nd round of reporter ion extraction
    dfQuan, reporterSummary = extractReporters(fracs, dfId, params, **reporterSummary)

    ###########################
    # TMT impurity correction #
    ###########################
    dfQuan = correctImpurity(dfQuan, params)

    #####################
    # Filtering of PSMs #
    #####################
    prot2psm = dfId.groupby('Protein')['key'].apply(list).to_dict()
    dfQuan = filterPSMs(dfQuan, prot2psm, params)

    #####################################
    # Show the loading-bias information #
    #####################################
    avgLb, sdLb, semLb, nn = getLoadingBias(dfQuan, params)
    print("  Loading bias (before correction)")
    print("  Reporter\tMean[%]\tSD[%]\tSEM[%]\t#PSMs")
    for i in range(len(reporters)):
        print("  %s\t%.2f\t%.2f\t%.2f\t%d" % (reporters[i], avgLb[i], sdLb[i], semLb[i], nn))

    #################
    # Normalization #
    #################
    dfNorm = normalization(dfQuan, params)

    
    ########################################################
    # save the PSM normalized intensities to output folder #
    #######################################################
    final_directory = os.path.join(os.getcwd(), params["save_dir"])
    os.makedirs(final_directory, exist_ok=True)
    outfile = open(os.path.join(final_directory, r"normalized_quan_psm_nonzero.txt"),"w") 
    dfQuan.index.name = "Fraction_scan"
    dfQuan.to_csv(outfile, sep = "\t", line_terminator='\n')
    outfile.close()

    ###############################
    # Protein-level summarization #
    ###############################
    dfProt = summarization(dfNorm, prot2psm, params)

    ###############################
    # Pepetide-level summarization #
    ###############################
    pep2psm = dfId.groupby('Peptide')['key'].apply(list).to_dict()
    dfPep = summarization(dfNorm, pep2psm, params)


	#########################
	# Generate output files #
	#########################
    
    outfile = open(os.path.join(final_directory, r"id_all_pep_quan.txt"),"w")
    outfile.write("All peptides quantified (n = %d)\n" %dfPep.shape[0] )
    dfPep.index.name = "Peptide"
    dfPep.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
    outfile.close()
    
    id_uni_pep = pd.read_table(os.path.dirname(params["idtxt"])+"/"+ 'publications/id_uni_pep.txt', sep="\t", skiprows=3, header=0, index_col = 0 )
    dfPep_uni = dfPep[dfPep.index.isin(id_uni_pep.index)]
    outfile = open(os.path.join(final_directory, r"id_uni_pep_quan.txt"),"w") 
    outfile.write("Unique peptides quantified (n = %d)\n" %dfPep_uni.shape[0] )
    dfPep_uni.index.name = "Peptide"
    dfPep_uni.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
    outfile.close()
    
    outfile = open(os.path.join(final_directory, r"id_all_prot_quan.txt"),"w") 
    outfile.write("All protein quantified (n = %d)\n" %dfProt.shape[0] )
    dfProt.index.name = "Protein Accession"
    dfProt.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
    outfile.close()
    
    id_uni_prot = pd.read_table(os.path.dirname(params["idtxt"])+"/"+ 'publications/id_uni_prot.txt', sep="\t", skiprows=1, header=0, index_col = 1)
    dfProt_uni = dfProt[dfProt.index.isin(id_uni_prot.index)]
    outfile = open(os.path.join(final_directory, r"id_uni_prot_quan.txt"),"w") 
    outfile.write("Unique protein quantified (n = %d)\n" %dfProt_uni.shape[0] )
    dfProt_uni.index.name = "Protein Accession"
    dfProt_uni.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
    outfile.close()
    
    print("\n  Completed generating output files \n\n  ***  jump-q program completed  ***\n")

