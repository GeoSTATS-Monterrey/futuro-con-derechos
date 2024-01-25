import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

"""
Libreria para encontrar el dispensador de alcohol mas cercano a una escuela

regresa las distancias e indices de los dispensadores mas cercanos a las escuelas

Teoricamente se puede usar para cualquier par de DF con un par de latitud, longitud


"""


def semiverseno(arr1, arr2):
    R = 6371  # Radio de la Tierra

    # Coso para que no se rompa 
    if arr1.ndim == 1:
        arr1 = arr1.reshape(1, -1)
    if arr2.ndim == 1:
        arr2 = arr2.reshape(1, -1)

    lat1, lon1 = np.radians(arr1[:, 0]), np.radians(arr1[:, 1])
    lat2, lon2 = np.radians(arr2[:, 0]), np.radians(arr2[:, 1])

    dlat = lat2.reshape(-1, 1) - lat1.reshape(1, -1)
    dlon = lon2.reshape(-1, 1) - lon1.reshape(1, -1)

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1).reshape(1, -1) * np.cos(lat2).reshape(-1, 1) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def find_nearest_n_places(df_escuelas, df_alcohol, n_neighbors=1):
    coords_df1 = df_escuelas[['latitud', 'longitud']].to_numpy()
    coords_df2 = df_alcohol[['latitud', 'longitud']].to_numpy()

    # Validation checks
    assert coords_df1.ndim == 2 and coords_df1.shape[1] == 2, "coords_df1 must be a 2D array with shape (n_samples, 2)"
    assert coords_df2.ndim == 2 and coords_df2.shape[1] == 2, "coords_df2 must be a 2D array with shape (n_samples, 2)"

    # Using NearestNeighbors from sklearn with haversine distance
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=semiverseno).fit(coords_df2)
    distances, indices = nbrs.kneighbors(coords_df1)

    return distances, indices


def find_nearest_within_radius(df_escuelas, df_alcohol, radius):
    # Identify valid and invalid rows
    valid_rows = df_escuelas[['latitud', 'longitud']].notna().all(axis=1)
    invalid_rows = ~valid_rows

    # Filter for valid rows
    valid_escuelas = df_escuelas[valid_rows]
    coords_df1 = valid_escuelas[['latitud', 'longitud']].to_numpy()
    coords_df2 = df_alcohol[['latitud', 'longitud']].to_numpy()

    # Perform nearest neighbor search on valid data
    nbrs = NearestNeighbors(radius=radius, metric=semiverseno).fit(coords_df2)
    valid_distances, valid_indices = nbrs.radius_neighbors(coords_df1)

    # Initialize results with placeholders
    all_distances = np.empty(len(df_escuelas), dtype=object)
    all_indices = np.empty(len(df_escuelas), dtype=object)
    all_distances[invalid_rows] = 0
    all_indices[invalid_rows] = [0]

    # Assign the calculated distances and indices to valid rows
    all_distances[valid_rows] = valid_distances
    all_indices[valid_rows] = valid_indices

    return all_distances, all_indices