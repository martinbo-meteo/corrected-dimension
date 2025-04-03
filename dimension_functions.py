import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm
import matplotlib.pyplot as plt

def usual_dim_formula(percentile, dist_analogs):
    mask = np.isfinite(dist_analogs)
    return 1/(np.mean(np.log(percentile/dist_analogs)[mask]))

def corrected_dim_formula(percentile, dist_analogs, densities):
    mask = np.isfinite(dist_analogs)
    return 1/np.average(np.log(percentile/dist_analogs)[mask], weights = 1/densities[mask])

def comp_radii_distances(pts, comp_pt, percentages, coord_axis = 1):
    '''
    computes the radii and the distances from the computation point comp_pt to all other points
    coord_axis describes the axis over which to compute the euclidean norm. If coord_axis is None, np.linalg.norm is replaced by np.abs
    '''

    if coord_axis is not None:
        dist = np.linalg.norm(pts - comp_pt, axis = coord_axis)
    else:
        dist = np.abs(pts - comp_pt)
        
    dist = np.where(dist == 0, np.inf, dist) # replace the 0 distance by np.inf, so that it is always ranked at the end
    radii = np.percentile(dist, percentages)

    return radii, dist
    
def comp_usual_dim(pts, comp_pt, percentages, coord_axis = 1):
    
    usual_dim = np.full(len(percentages), np.nan)
    
    radii, dist = comp_radii_distances(pts, comp_pt, percentages, coord_axis = coord_axis)
    
    for j in range(len(percentages)):
        dist_analogs = dist[dist < radii[j]]
        usual_dim[j] = usual_dim_formula(radii[j], dist_analogs)

    return radii, usual_dim

def comp_corrected_dim(pts, comp_pt, percentages, bandwidths, provided_kde = None, verbose = True, coord_axis = 1):
    
    corrected_dim = np.full((len(percentages), len(bandwidths)), np.nan)
    
    radii, dist = comp_radii_distances(pts, comp_pt, percentages, coord_axis = coord_axis)
    
    for j in tqdm(range(len(percentages)), disable = not verbose):
        for k, bw in enumerate(bandwidths):
            
            if provided_kde is None:
                # construct the density including the maximum number of points
                analogs_max = pts[dist < np.max(radii)]
                kde = gaussian_kde(analogs_max.T, bw_method = bw)
            else:
                kde = provided_kde
                
            # compute the corrected dimension, with the weights being the inverse of the density
            analogs = pts[dist < radii[j]]
            dist_analogs = dist[dist < radii[j]]
            densities = kde.evaluate(analogs.T)
            corrected_dim[j, k] = corrected_dim_formula(radii[j], dist_analogs, densities)

    # squeeze in case there is only one bandwidth and the second dimension of the arrays is 1
    return radii, corrected_dim.squeeze()

def comp_dim_resampling(pts, comp_pt, percentages, new_samples, bandwidths, coord_axis = 1, verbose = True, corrected = False):
    
    # compute first which of the points are close enough to comp_pt
    radii, dist = comp_radii_distances(pts, comp_pt, np.max(percentages), coord_axis = coord_axis)
    analogs = pts[dist < np.max(radii)]

    dim = np.full((len(percentages), len(bandwidths)), np.nan)
    resampled_radii = np.full((len(percentages), len(bandwidths)), np.nan)
    
    for k, bw in enumerate(tqdm(bandwidths)):
        
        # use these points to estimate the density around comp_pt
        kde = gaussian_kde(analogs.T, bw_method = bw)

        # draw new analogs
        new_analogs = kde.resample(new_samples).T

        # I want this method to produce the same radii; this will be the case if I adapt the percentages accordingly
        new_percentages = percentages/np.max(percentages) * 100
        
        if corrected == True:
            resampled_radii[:, k], dim[:, k] = comp_corrected_dim(new_analogs, comp_pt, new_percentages, [bw], provided_kde = kde, verbose = True, coord_axis = coord_axis)
        else:
            resampled_radii[:, k], dim[:, k] = comp_usual_dim(new_analogs, comp_pt, new_percentages, coord_axis = coord_axis)
            
    return resampled_radii, dim

def plot_dim(radii, dim, n_kept, **kwargs):
    '''function to plot the dimension, with lighter line when the number of points used in the fit is smaller than 400'''
    
    epi = np.argwhere(n_kept > 400).squeeze()[0] # Enough Point Index
    
    plt.plot(radii[epi:], dim[epi:], **kwargs)

    # in order to put only once the label
    if 'label' in kwargs:
        kwargs.pop('label')
        
    plt.plot(radii[:epi + 1], dim[:epi + 1], alpha = 0.2, **kwargs)

    plt.xlabel('$R$')
    plt.ylabel('dimension')
    

        