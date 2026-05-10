"""
Script de Data Preparation (ETL)
Responsável por extrair os dados crus, realizar as transformações, engenharia de features espaciais
e salvar as bases finais prontas para modelagem.
"""

import sys
import os
import pandas as pd
import geopandas as gpd
import numpy as np

# Adicionar a pasta raiz ao sys.path para importar o pacote utils (voltar uma pasta a partir de 'python/')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_cleaning import trata_valor, estimate_rented_days
from utils.geospatial import get_rio_pois, get_nearest_neighbor_info
from utils.quadtree import QuadPoint, Boundary, QuadTree, get_params

def main():
    print("Iniciando Data Preparation Pipeline...")

    # --- 1. CONFIGURAÇÃO DE CAMINHOS ---
    path_raw = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    path_processed = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
    os.makedirs(path_processed, exist_ok=True)

    # --- 2. CARGA DE DADOS CRUS (LOAD) ---
    print("\n[1/7] Carregando dados crus...")
    df_anuncios = pd.read_excel(os.path.join(path_raw, 'raw_data_quintoandar.xlsx'), index_col=0)
    df_listings = pd.read_csv(os.path.join(path_raw, 'raw_data_airbnb.csv'))
    
    # Shapefiles
    df_setores_censitarios = gpd.read_file(os.path.join(path_raw, 'shapefile_rio', 'Setores_Censitarios_2022.shp'))
    df_costa_maritima = gpd.read_file(os.path.join(path_raw, 'world_countries_coasts', 'world_countries_coasts.shp'))
    geometria_maritima = df_costa_maritima.geometry.iloc[0]
    
    # POIs (Pontos de Interesse via API)
    print("Baixando Pontos de Interesse (POIs) do RJ via Overpass API...")
    df_pois = get_rio_pois()

    # --- 3. LIMPEZA BÁSICA (CLEANING) ---
    print("\n[2/7] Aplicando limpeza de dados básica...")
    # Airbnb
    df_listings = estimate_rented_days(df_listings)
    df_listings['price'] = df_listings['price'].apply(trata_valor)
    df_listings = df_listings.dropna(subset=['price']).reset_index(drop=True)

    # --- 4. JUNÇÃO ESPACIAL BÁSICA (SPATIAL JOINS) ---
    print("\n[3/7] Realizando Joins Espaciais (Setores Censitários e Bairros)...")
    # Quinto Andar
    df_anuncios_cleaned = df_anuncios.dropna(subset=['latitude', 'longitude'])
    gdf_anuncios = gpd.GeoDataFrame(
        df_anuncios_cleaned,
        geometry=gpd.points_from_xy(df_anuncios_cleaned.longitude, df_anuncios_cleaned.latitude),
        crs="WGS84"
    ).to_crs(df_setores_censitarios.crs)
    
    anuncios_geoloc = gpd.sjoin(gdf_anuncios, df_setores_censitarios[['geometry','cd_bairro','nm_bairro']], how="inner", predicate="within")

    # Airbnb
    df_listings_cleaned = df_listings.dropna(subset=['latitude', 'longitude'])
    gdf_listings = gpd.GeoDataFrame(
        df_listings_cleaned,
        geometry=gpd.points_from_xy(df_listings_cleaned.longitude, df_listings_cleaned.latitude),
        crs="WGS84"
    ).to_crs(df_setores_censitarios.crs)

    listings_geoloc = gpd.sjoin(gdf_listings, df_setores_censitarios[['geometry','cd_bairro','nm_bairro']], how="inner", predicate="within")

    # --- 5. ENGENHARIA DE FEATURES GEOGRÁFICAS (DISTÂNCIAS) ---
    print("\n[4/7] Calculando distâncias para a praia e POIs...")
    
    # Distância Praia
    geometria_maritima_gs = gpd.GeoSeries([geometria_maritima], crs="EPSG:4326")
    
    anuncios_points = gpd.GeoSeries(gpd.points_from_xy(anuncios_geoloc.longitude, anuncios_geoloc.latitude), crs="EPSG:4326")
    anuncios_geoloc['distancia_minima_praia'] = anuncios_points.to_crs(anuncios_geoloc.crs).distance(geometria_maritima_gs.to_crs(anuncios_geoloc.crs).iloc[0])

    listings_points = gpd.GeoSeries(gpd.points_from_xy(listings_geoloc.longitude, listings_geoloc.latitude), crs="EPSG:4326")
    listings_geoloc['distancia_minima_praia'] = listings_points.to_crs(listings_geoloc.crs).distance(geometria_maritima_gs.to_crs(listings_geoloc.crs).iloc[0])

    # Distância POIs (KNN - BallTree)
    tipos_contemplados = ['attraction','subway_entrance', 'school']
    for t in tipos_contemplados:
        df_target = df_pois[df_pois['tipo']==t].reset_index(drop=True)
        if not df_target.empty:
            # Anuncios
            nearest_info_anuncios = get_nearest_neighbor_info(anuncios_geoloc, df_target)
            anuncios_geoloc[f'nome_{t}_mais_proximo'] = df_target.iloc[nearest_info_anuncios['nearest_index'].values]['nome'].values
            anuncios_geoloc[f'distancia_minima_{t}'] = nearest_info_anuncios['distance_km'].values
            
            # Listings
            nearest_info_listings = get_nearest_neighbor_info(listings_geoloc, df_target)
            listings_geoloc[f'nome_{t}_mais_proximo'] = df_target.iloc[nearest_info_listings['nearest_index'].values]['nome'].values
            listings_geoloc[f'distancia_minima_{t}'] = nearest_info_listings['distance_km'].values

    # --- 6. PADRONIZAÇÃO DE NOMES E ESCALAS LOGARÍTMICAS ---
    print("\n[5/7] Padronizando nomenclaturas e aplicando escalas Logarítmicas...")
    
    # Anuncios QuintoAndar
    rename_anuncios = {
        'latitude': 'Latitude', 'longitude': 'Longitude', 'hasFurniture': 'Mobiliado',
        'salePrice': 'Valor de venda', 'area': 'Área construída', 'bathrooms': 'Banheiros',
        'bedrooms': 'Quartos', 'parkingSpaces': 'Vagas', 'nm_bairro': 'Bairro', 'street': 'Logradouro',
        'condoPrice': 'Valor do condominio', 'distancia_minima_praia': 'Distancia da praia',
        'distancia_minima_attraction': 'Distância minima de atracao turistica',
        'distancia_minima_subway_entrance': 'Distância minima de entrada de metrô',
        'distancia_minima_school': 'Distância minima de escola'
    }
    anuncios_geoloc.rename(columns=rename_anuncios, inplace=True)
    
    anuncios_geoloc['Valor de venda (escala log)'] = anuncios_geoloc['Valor de venda'].map(np.log1p)
    anuncios_geoloc['Valor do condominio (escala log)'] = anuncios_geoloc['Valor do condominio'].map(np.log1p)
    anuncios_geoloc['Área construída (escala log)'] = anuncios_geoloc['Área construída'].map(np.log1p)

    targetlog_anuncios = 'Valor de venda (escala log)'

    # Listings Airbnb
    rename_listings = {
        'latitude': 'Latitude', 'longitude': 'Longitude', 'room_type': 'Tipo de locação',
        'nm_bairro':'Bairro', 'price': 'Preço do aluguel', 'bathrooms':'Banheiros',
        'bedrooms': 'Quartos', 'beds': 'Camas', 'accommodates': 'Qtd de pessoas acomodadas',
        'minimum_nights': 'Mínimo de noites', 'number_of_reviews': 'Número de reviews',
        'reviews_per_month': 'Reviews por mês', 'calculated_host_listings_count': 'Contagem de hosts',
        'availability_365': 'Disponibilidade no ano', 'distancia_minima_praia': 'Distancia da praia',
        'distancia_minima_attraction': 'Distância minima de atracao turistica',
        'distancia_minima_subway_entrance': 'Distância minima de entrada de metrô',
        'distancia_minima_school': 'Distância minima de escola',
        'estimated_days_rented_ltm': 'Dias alugados (estimado)'
    }
    listings_geoloc.rename(columns=rename_listings, inplace=True)
    
    listings_geoloc['Preço do aluguel (escala log)'] = listings_geoloc['Preço do aluguel'].map(np.log1p)
    targetlog_listings = 'Preço do aluguel (escala log)'

    # --- 7. CLUSTERIZAÇÃO ESPACIAL (QUADTREE) ---
    print("\n[6/7] Processando QuadTrees Espaciais...")
    lat_rj, lon_rj = (-22.7037, -23.1856), (-43.0846, -43.8086)
    c_lon, c_lat, w, h = get_params(lat_rj, lon_rj)

    # Anuncios
    qt_anuncios = QuadTree(Boundary(c_lon, c_lat, w, h), capacity=500)
    for _, row in anuncios_geoloc.iterrows():
        qt_anuncios.insert(QuadPoint(row['Longitude'], row['Latitude'], row[targetlog_anuncios]))
    anuncios_geoloc['quadrante'] = anuncios_geoloc.apply(
        lambda row: qt_anuncios.get_quadrant_id(QuadPoint(row['Longitude'], row['Latitude'], None)), axis=1)
    anuncios_geoloc['poligono_quadrante'] = anuncios_geoloc.apply(
        lambda row: qt_anuncios.get_quadrant_geometry(row['quadrante']), axis=1)

    # Listings
    qt_listings = QuadTree(Boundary(c_lon, c_lat, w, h), capacity=100)
    for _, row in listings_geoloc.iterrows():
        qt_listings.insert(QuadPoint(row['Longitude'], row['Latitude'], row[targetlog_listings]))
    listings_geoloc['quadrante'] = listings_geoloc.apply(
        lambda row: qt_listings.get_quadrant_id(QuadPoint(row['Longitude'], row['Latitude'], None)), axis=1)
    listings_geoloc['poligono_quadrante'] = listings_geoloc.apply(
        lambda row: qt_listings.get_quadrant_geometry(row['quadrante']), axis=1)

    # --- 8. EXPORTAÇÃO (SAVE) ---
    print("\n[7/7] Salvando datasets processados como CSV...")
    # Convertendo a coluna de geometria para WKT (padrão) para salvar em CSV, ou dropando ela se não for necessária.
    # O GeoPandas mantém a coluna 'geometry'.
    
    anuncios_geoloc.to_csv(os.path.join(path_processed, 'base_features_quintoAndar.csv'), index=False)
    listings_geoloc.to_csv(os.path.join(path_processed, 'base_features_AirBnb.csv'), index=False)

    print(f"\n✅ Data Preparation finalizado com sucesso! Arquivos salvos em: {path_processed}")

if __name__ == "__main__":
    main()
