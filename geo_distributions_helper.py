import numpy as np
import pandas as pd
from pyproj import Geod
import ot

geod = Geod(ellps="WGS84")


def build_distributions(df):
    lang_distributions = {}
    for iso, grp in df.groupby("glottocode"):
        centroid = grp[["Centroid_Lon","Centroid_Lat"]].to_numpy(dtype="f8")
        weights   = grp["weight"].to_numpy(dtype="f8")
        lang_distributions[iso] = (centroid, weights)
    return lang_distributions

def integrate_geo_data(language_centroid_style):
    # print("HELLO RUNNING INTEGRATE_GEO_DATA")
    df = {}
    if (language_centroid_style == 0):
        path = "data/language_country_merged_data_old_coord.csv"
        df = pd.read_csv(path)
    elif (language_centroid_style == 1):
        path = "data/language_country_merged_data_new_coord.csv"
        df = pd.read_csv(path)
    else:
        print("Not Valid Language Centroid Style Input (Please Choose Either 0 or 1)")
        
    # ensure the 4 necessary columns exist
    expected = {"glottocode","Centroid_Lon","Centroid_Lat","weight"}
    if not expected.issubset(df.columns):
        missing = expected - set(df.columns)
        raise ValueError(f"language_country_merged_data_new/old_coord.csv is missing columns: {missing}")

    geo_distributions = build_distributions(df)
    return geo_distributions

def geodesic_distance(lon1, lat1, lon2, lat2):
    az12, az21, dist = geod.inv(lon1, lat1, lon2, lat2, radians = False)
    return dist/1000

DISTANCE_MAX_1 = geodesic_distance(0, 90, 0, -90)
DISTANCE_MAX_2 = 19999.696911518036 #This is used for a second normalization tactic

def cost_matrix_calculate(iso1, iso2, lang_distributions):
    ptsA, w1 = lang_distributions[iso1]
    ptsB, w2 = lang_distributions[iso2]
    lonA, latA = ptsA[:, 0], ptsA[:, 1]
    lonB, latB = ptsB[:, 0], ptsB[:, 1]

    cost_matrix = np.zeros((len(lonA), len(lonB)))

    for i in range(len(lonA)):
        for j in range(len(lonB)):
            cost_matrix[i, j] = geodesic_distance(lonA[i], latA[i], lonB[j], latB[j])

    return cost_matrix

def w1_distance_language(iso1, iso2, lang_distributions):
    cost_matrix = cost_matrix_calculate(iso1, iso2, lang_distributions)

    ptsA, wA = lang_distributions[iso1]
    ptsB, wB = lang_distributions[iso2]

    return ot.emd2(wA, wB, cost_matrix)


def normalized_w1_distance(iso1, iso2, lang_distributions, normalize_type):
    lang_dist = w1_distance_language(iso1, iso2, lang_distributions)
    if (normalize_type == 1):
        return lang_dist / DISTANCE_MAX_1
    elif (normalize_type == 2):
        return lang_dist/ DISTANCE_MAX_2
    else:
        print("Error: You Needed to pass in a valid value (1 or 2) for your normalizing type")