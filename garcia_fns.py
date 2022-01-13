import numpy as np
import pandas as pd
import scipy
from scipy import fft
import copy

def garcia_robust_fitting(NDVI_profile, NDVIhat, r_weights, d_eigs,
                          n, n_miss, neg_res_only=False,Sopt_Rog_val=False,Sopt_range=False):

    DCT = fft.dct((r_weights*(NDVI_profile - NDVIhat)
                         + NDVIhat),
                        norm='ortho')

    # Step 4 - Loop S
    if not Sopt_Rog_val:
        if Sopt_range is False:
            smoothing = 10**np.arange(-3,4.2,0.2)
        else:
          smoothing = 10**Sopt_range
    else: smoothing = [Sopt_Rog_val]
    gcv_temp = []
    tempNDVI_arr = []
    for s in smoothing:

        gcv, NDVIhat = GCV_score(NDVI_profile, NDVI_DCT=DCT,
                                 r_weights=r_weights, s_val=s,
                                 d_eigs=d_eigs, n=n, n_miss=n_miss)
        
        gcv_temp.append(gcv)
        tempNDVI_arr.append(NDVIhat)

    gcv_temp = np.array(gcv_temp)
    best_gcv = gcv_temp[gcv_temp[:,0].argmin()]

    s = gcv_temp[gcv_temp[:,0].argmin()][1]
    gamma = 1 / (1 + s*((-1*d_eigs)**2))
    r_arr = NDVI_profile - tempNDVI_arr[gcv_temp[:,0].argmin()]

    MAD = np.median(np.abs(r_arr[r_weights!=0]
                        - np.median(r_arr[r_weights!=0])))
    u_arr = (r_arr / (1.4826 * MAD * np.sqrt(1-gamma.sum()/n)))
    
    r_weighting = (1 - (u_arr/4.685)**2)**2
    r_weighting[(np.abs(u_arr/4.685)>1)] = 0

    if neg_res_only is True:

        r_weighting[r_arr > 0] = 1

    return(tempNDVI_arr[gcv_temp[:,0].argmin()],
           r_weighting, best_gcv, gcv_temp)
           
           
def GCV_score(NDVI_profile, NDVI_DCT, r_weights, s_val,
              d_eigs, n, n_miss):

    gamma = 1 / (1 + s_val*((-1*d_eigs)**2))

    NDVI_smoothed = fft.idct(gamma * NDVI_DCT,
                                norm='ortho')
    
    tr_H = gamma.sum()
    wsse =  (((r_weights**0.5)*(NDVI_profile-NDVI_smoothed))**2).sum()
    denominator = ((n-n_miss) * ((1-tr_H/n)**2))
    gcv_score = wsse/denominator

    return ([gcv_score,s_val], NDVI_smoothed)
    

def Garcia_smoothing_complete(NDVI_profile, fit_robust=False,
                              fit_envelope=False, neg_residuals_only=False,
                              Sopt_Rog=False,Sopt_range=False):
    # Copy of original NDVI profile
    _NDVI = copy.deepcopy(NDVI_profile)

    # Weights for missing values
    # weights = (_NDVI!=-3000).astype(int)
    weights = ((_NDVI!=-3000) & (_NDVI!=np.nan)).astype(int)
    # Initial robust weights (just ones so that it makes no difference)
    r_weights = np.ones(weights.shape)
    # Weighted NDVI profile
    NDVIhat = copy.deepcopy(NDVI_profile) * weights * r_weights
    
    # Length of profile 
    n = _NDVI.shape[0]
    # Number of missing values
    n_miss = n-weights.sum()

    # Eigenvalues
    d_eigs = -2+2*np.cos(np.arange(n)*np.pi/n)
    
    # Setting number of robust iterations to perform
    if not fit_robust: r_its = 1
    else: r_its = 5

    # Initialising lists for writing to
    robust_gcv = []
    robust_all_gcv = []
    robust_weights = [weights]


    # Robust loop
    for r_it in range(r_its):
        if (Sopt_Rog) & (r_it > 1):
            Sopt_Rog_val = robust_gcv[1][1]
        else: Sopt_Rog_val = False
        # Calling robust fitting function.
        results_garcia = garcia_robust_fitting(copy.deepcopy(_NDVI),
                                                NDVIhat=NDVIhat,
                                                r_weights=weights * r_weights,
                                                d_eigs=d_eigs,
                                                n=n,
                                                n_miss=n_miss,
                                                neg_res_only=neg_residuals_only,
                                                Sopt_Rog_val=Sopt_Rog_val,
                                                Sopt_range=Sopt_range)
        NDVIhat, r_weights, best_gcv, all_gcv = results_garcia
        robust_weights.append(weights * r_weights)
        robust_gcv.append(best_gcv)
        robust_all_gcv.append(all_gcv)
    
    robust_gcv = np.array(robust_gcv)
    # robust_all_gcv = np.array(robust_all_gcv)
    robust_weights = np.array(robust_weights)

    # Setting number of envelope iterations to perform
    if not fit_envelope:
        return(NDVIhat, robust_gcv, robust_weights)
    else: env_its = 4

    if Sopt_Rog: Sopt = robust_gcv[1,1]
    else: Sopt = robust_gcv[robust_gcv[:,0].argmin(),1]

    Wopt = robust_weights[-1]
    
    # Envelope Loop
    for env_it in range(env_its):
        DCT = fft.dct((Wopt*(_NDVI - NDVIhat)
                         + NDVIhat),
                        norm='ortho')
        gcv, NDVIhat = GCV_score(_NDVI, NDVI_DCT=DCT,
                                 r_weights=Wopt, s_val=Sopt,
                                 d_eigs=d_eigs, n=n, n_miss=n_miss)

        _NDVI[_NDVI<NDVIhat] = NDVIhat[_NDVI<NDVIhat]
        # NDVIhat = copy.deepcopy(_NDVI)

    return(NDVIhat, robust_gcv, robust_weights, robust_all_gcv)
