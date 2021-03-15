from typing import Tuple

import numpy as np


def compute_feature_distances(features1: np.ndarray, 
                              features2: np.ndarray) -> np.ndarray:
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """
    #broadcasting trick
    a = features1[:, np.newaxis, :]
    b = features2[np.newaxis, :, :]
    
    return np.linalg.norm( (a-b), axis=-1)


def match_features(features1: np.ndarray, 
                   features2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform nearest-neighbor matching with ratio test.

    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    The results should be sorted in descending order of confidence.

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)


    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    
    distances = compute_feature_distances(features1, features2)
    matches = np.empty((0,2), dtype=np.int8)
    confidences = np.array([])

    for i in range(0, distances.shape[0]):
      srted = np.argsort(distances[i])
      
      ratio = distances[i][srted[0]] / distances[i][srted[1]]
      if ( ratio < 0.84):
        matches = np.vstack((matches, [i, srted[0]]))
        confidences = np.append(confidences, ( distances[i][srted[1]] / distances[i][srted[0]]) )
    
    confidence_index = (-confidences).argsort(axis=0)
    sorted_matches = matches[confidence_index]
    sorted_confidences = confidences[confidence_index]
    
    return sorted_matches, sorted_confidences
