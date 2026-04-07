import itertools
import numpy as np

def compute_gromov_hyperbolicity(dist_matrix):
    """
    Compute Gromov hyperbolicity from a distance matrix.
    """
    n = dist_matrix.shape[0]
    deltas = []

    for i, j, k, l in itertools.combinations(range(n), 4):
        d01 = dist_matrix[i, j]
        d23 = dist_matrix[k, l]
        d02 = dist_matrix[i, k]
        d13 = dist_matrix[j, l]
        d03 = dist_matrix[i, l]
        d12 = dist_matrix[j, k]

        s = [d01 + d23, d02 + d13, d03 + d12]
        s.sort(reverse=True)

        delta = (s[0] - s[1]) / 2.0
        deltas.append(delta)
    print(deltas)

    return np.max(deltas), np.mean(deltas)




def compute_gromov_hyperbolicity_not_optimized(dist_matrix):
    """
    Compute Gromov hyperbolicity from a distance matrix.
    """
    n = dist_matrix.shape[0]
    deltas = []
    
    count = 0
    for i, j, k, l in itertools.permutations(range(n), 4):
        d01 = dist_matrix[i, j]
        d23 = dist_matrix[k, l]
        d02 = dist_matrix[i, k]
        d13 = dist_matrix[j, l]
        d03 = dist_matrix[i, l]
        d12 = dist_matrix[j, k]

        s = [d01 + d23, d02 + d13, d03 + d12]
        
        
        delta = 0.5 * (s[0] - max(s[1], s[2]))

        deltas.append(delta)
   
    print(count)
    return np.max(deltas), np.mean(deltas)