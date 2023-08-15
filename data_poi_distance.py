import pandas as pd
import numpy as np


def _get_distance_array(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    h_dist = c * r

    return h_dist


city = 'NYC'
df_poi_mapping = pd.read_csv(f"./raw_data/{city}_poi_mapping.csv")

# Get latitude and longitude
latitude = df_poi_mapping['latitude'].values
longitude = df_poi_mapping['longitude'].values

# Calculate distance and construct distance table
df_distance = pd.DataFrame(_get_distance_array(latitude[:, np.newaxis], longitude[:, np.newaxis], latitude, longitude))

# Create a empty DataFrame to store the farthest 100 POIs
df_farthest = pd.DataFrame(index=df_distance.index, columns=range(100))

for poi in df_distance.index:
    # For each POI, sort by distance in descending order
    sorted_distances = df_distance.loc[poi].sort_values(ascending=False)
    # Get the index of the farthest 100 POIs
    farthest_POIs = sorted_distances.index[:100]
    # Save the result to df_farthest
    df_farthest.loc[poi] = farthest_POIs

df_farthest.to_csv(f'./raw_data/{city}_farthest_POIs.csv', index=False)
