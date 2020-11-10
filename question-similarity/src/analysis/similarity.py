from scipy.spatial.distance import cosine
import numpy as np

# return cosine distance between 2 vectors
def calculate_cosine_distance(u: np.array, v: np.array):
    return cosine(u, v)

