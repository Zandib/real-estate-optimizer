import pandas as pd
import numpy as np
import requests
from requests.exceptions import RequestException, JSONDecodeError
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed
import multiprocessing

def get_rio_pois():
    """
    Busca Pontos de Interesse (Escolas, Metrô, Turismo) no RJ usando a Overpass API.
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json];
    area["name"="Rio de Janeiro"]->.searchArea;
    (
      node["amenity"="school"](area.searchArea);
      node["railway"="subway_entrance"](area.searchArea);
      node["tourism"](area.searchArea);
    );
    out body;
    """
    headers = {
        'User-Agent': 'LocalNotebook/1.0'
    }
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, headers=headers)
        response.raise_for_status()
        data = response.json()
    except JSONDecodeError as e:
        print(f"JSON Decode Error from Overpass API: {e}")
        return pd.DataFrame()
    except RequestException as e:
        print(f"Request Error to Overpass API: {e}")
        return pd.DataFrame()

    pois = []
    for element in data['elements']:
        tags = element.get('tags', {})
        tipo = 'N/A'
        if 'amenity' in tags:
            tipo = tags['amenity']
        elif 'railway' in tags:
            tipo = tags['railway']
        elif 'tourism' in tags:
            tipo = tags['tourism']

        pois.append({
            'latitude': element['lat'],
            'longitude': element['lon'],
            'nome': tags.get('name', 'N/A'),
            'tipo': tipo
        })

    return pd.DataFrame(pois)

def get_nearest_neighbor_info(df_source, df_target, lat_col='latitude', long_col='longitude', n_jobs=-1):
    """
    Retorna um DataFrame com o índice do vizinho mais próximo e a
    distância de Haversine (em km) para cada ponto de df_source.
    """
    src_rad = np.deg2rad(df_source[[lat_col, long_col]].values)
    tgt_rad = np.deg2rad(df_target[['latitude', 'longitude']].values)

    tree = BallTree(tgt_rad, metric='haversine')

    def query_chunk(chunk):
        dist, ind = tree.query(chunk, k=1)
        return dist.flatten(), ind.flatten()

    num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    chunks = np.array_split(src_rad, num_cores)

    results = Parallel(n_jobs=num_cores)(
        delayed(query_chunk)(chunk) for chunk in chunks
    )
    distances_rad, positions = zip(*results)

    final_distances = np.concatenate(distances_rad) * 6371
    final_positions = np.concatenate(positions)

    return pd.DataFrame({
        'nearest_index': df_target.index[final_positions],
        'distance_km': final_distances
    }, index=df_source.index)
