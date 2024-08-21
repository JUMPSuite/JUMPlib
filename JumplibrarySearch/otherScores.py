import scipy
import pandas as pd
import numpy as np
import scipy.stats
#p = query numpy array
#q = lirary numpy array
def DP_Peng_similarity(p, q):
    return np.dot(np.sqrt(p),np.sqrt(q))


#Clustering millions of tandem mass spectra, J Proteome Res. 2008; 7: 113-22
#JUMPm --- dot product calculation
def normalizedDotProduct(p, q):
    num = np.dot(np.sqrt(p),np.sqrt(q))
    den1 = np.sum(p*p)
    den2 = np.sum(q*q)
    if den1 * den2 == 0:
        normDotProduct = 0
    else:
        normDotProduct = num/np.sqrt(den1*den2)
    return normDotProduct

# def normalizedDotProduct(e_intens,r_intens):
    
#     if np.sqrt( sum(e_intens*e_intens)*sum(r_intens*r_intens) )==0.0:
#         e_sim=0.0
#     else:
#         e_sim = sum(e_intens*r_intens)/np.sqrt( sum(e_intens*e_intens)*sum(r_intens*r_intens) )
    
#     return e_sim

#This function assumes that the preprocessing between spectra is already done and the feat spec and lib spec has matched ions
#the ions that were missing in featSpec but present in libSpec has intensity 0
#However, library is always constant
#Also mz column in feat spectra is already replaced by library m/z values as they are within the tolerance 
def unweightedEntropySimCalc(featSpec,libSpec):
    df = pd.DataFrame(featSpec)
    df.columns = ["mz","p"] #rename columns
    #add another column with library intensity (no need to add mz column as mz value is identical)
    df["q"] = libSpec[:,1]
    #numpy array are added to get the merged column
    df["merged"] = df["p"]+df["q"]
    
    #entropy increase is computed 
    
    r"""
    Unweighted entropy distance:

    .. math::

        -\frac{2\times S_{PQ}-S_P-S_Q} {ln(4)}, S_I=\sum_{i} {I_i ln(I_i)}
    """
    
    
    entropy_increase = 2 * scipy.stats.entropy(df.merged) - scipy.stats.entropy(df.p) - scipy.stats.entropy(df.q)
    similarity = 1-(entropy_increase/np.log(4))
    #if similarity > 1:
    #    similarity =1
    #if similarity < 0:
    #    similarity = 0
    return similarity


#This function converts the dictionary form spectra to numpy array with position 0 being mz and position 1 being intensity
def conversionDictSpecToNumpyArrayFormat(spectra):
    numpyArray=[]
    mz = spectra["mz"]
    intensity = spectra["intensity"]
    for index, value in enumerate(mz):
        numpyArray.append([value,intensity[index]])
    return np.array(numpyArray, dtype=np.float32)


def fidelity_similarity(p, q):
    r"""
    Fidelity distance:

    .. math::

        1-\sum\sqrt{P_{i}Q_{i}}
    """
    #p_ = p/np.sum(p)
    #q_ = q/np.sum(q)
    
    #return np.sum(np.sqrt(p_ * q_))
    return np.sum(np.sqrt(p*q))

def dot_product_similarity(p, q):
    r"""
    Dot-Product distance:

    .. math::

        1 - (\frac{(\sum{Q_iP_i})^2}{\sum{Q_i^2\sum P_i^2}})^1/2
    """
    #p_ = p/np.sum(p)
    #q_ = q/np.sum(q)
    
    score = np.power(np.sum(q * p), 2) / \
            (np.sum(np.power(q, 2)) * np.sum(np.power(p, 2)))
    return score



def bhattacharya_2_similarity(p, q):
    r"""
    Bhattacharya 2 distance:

    .. math::

        -\ln{(\sum\sqrt{P_{i}Q_{i}})}
    """
    #p_ = p/np.sum(p)
    #q_ = q/np.sum(q)
    s = np.sum(np.sqrt(p * q))
    den = 1-np.log(s)
    similarity = 1/den
    
    return similarity



