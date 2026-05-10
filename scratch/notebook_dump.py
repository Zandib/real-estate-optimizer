# ----- CELL 1 -----
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from unidecode import unidecode
import re

pd.set_option('display.max_columns',999)

# ----- CELL 4 -----

path_raw = '../data/raw/'
path_processed = '../data/processed/'

# ----- CELL 6 -----
df_anuncios = pd.read_excel(path_raw+'raw_data_quintoandar.xlsx',index_col=0)

# ----- CELL 7 -----
df_anuncios.head()

# ----- CELL 9 -----
df_listings = pd.read_csv(path_raw+'raw_data_airbnb.csv')

# ----- CELL 10 -----
def trata_valor(x):
  x = str(x).replace('$','').replace(',','')
  return float(x)

def estimate_rented_days(df):
  REVIEW_RATE = 0.5  # Assume-se que 50% dos hóspedes deixam reviews
  DEFAULT_STAY = 3   # Média de noites caso o minimum_nights seja irreal

  # Criamos uma cópia para não alterar o original acidentalmente
  df_calc = df.copy()

  # 2. Tratamento do minimum_nights
  # Limpamos valores extremos (ex: estadias mínimas de 1 ano) para não distorcer a média
  # É comum travar o teto em 30 dias para análise de aluguel de temporada
  df_calc['clamped_min_nights'] = df_calc['minimum_nights'].clip(upper=30)

  # 3. Estimativa de Reservas Totais (Total Bookings)
  # Bookings = Reviews / Taxa de Review
  df_calc['estimated_bookings'] = df_calc['number_of_reviews'] / REVIEW_RATE

  # 4. Cálculo da Coluna Alvo: Dias Alugados (Estimado)
  # Dias = Reservas * Estadia Mínima
  df_calc['estimated_days_rented'] = df_calc['estimated_bookings'] * df_calc['clamped_min_nights']

  # 5. Estimativa Anual (opcional, baseada nos últimos 12 meses)
  # Útil se você estiver analisando rendimento anual
  df_calc['estimated_days_rented_ltm'] = (df_calc['number_of_reviews_ltm'] / REVIEW_RATE) * df_calc['clamped_min_nights']

  # Garantir que não passamos de 365 dias na métrica anual
  df_calc['estimated_days_rented_ltm'] = df_calc['estimated_days_rented_ltm'].clip(upper=365)

  return df_calc

# ----- CELL 11 -----
df_listings = estimate_rented_days(df_listings)

# ----- CELL 12 -----
df_listings['price'] = df_listings['price'].apply(trata_valor)
df_listings = df_listings.dropna(subset=['price']).reset_index(drop=True)

# ----- CELL 13 -----
df_listings.head()

# ----- CELL 14 -----
df_neighbourhoods = pd.read_csv(path_raw+'neighbourhoods.csv')

# ----- CELL 15 -----
df_neighbourhoods.head()

# ----- CELL 16 -----
df_neighbourhoods['neighbourhood'].unique()

# ----- CELL 17 -----
df_neighbourhoods.shape

# ----- CELL 18 -----
df_reviews = pd.read_csv(path_raw+'reviews_completa.csv')

# ----- CELL 19 -----
df_reviews.head(25)

# ----- CELL 21 -----
from PIL import Image

image_path = path_raw + 'mapa_rio.webp'
img = Image.open(image_path)

# You can display the image if needed (uncomment the line below)
# img.show()
print(f"Image loaded successfully. Format: {img.format}, Size: {img.size}, Mode: {img.mode}")

# ----- CELL 22 -----
import geopandas as gpd

shapefile_path = path_raw + 'shapefile_rio/Setores_Censitarios_2022.shp'
df_setores_censitarios = gpd.read_file(shapefile_path)

print(df_setores_censitarios.columns)
print(f"Shape of df_setores_censitarios: {df_setores_censitarios.shape}")
df_setores_censitarios.head()

# ----- CELL 24 -----
import geopandas as gpd

shapefile_costa_path = path_raw + 'world_countries_coasts/world_countries_coasts.shp'
df_costa_maritima = gpd.read_file(shapefile_costa_path)

print(f"Shape of df_costa_maritima: {df_costa_maritima.shape}")
df_costa_maritima.head()

# ----- CELL 25 -----
geometria_maritima = df_costa_maritima.geometry.iloc[0]

# ----- CELL 26 -----
geometria_maritima

# ----- CELL 28 -----
import pandas as pd
import requests
from requests.exceptions import RequestException, JSONDecodeError

def get_rio_pois():
    overpass_url = "http://overpass-api.de/api/interpreter"
    # Query para buscar estações de metrô, escolas e pontos turísticos
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
        'User-Agent': 'ColabNotebook/1.0 (https://colab.research.google.com)' # Adiciona o cabeçalho User-Agent
    }
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, headers=headers)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
    except JSONDecodeError as e:
        print(f"JSON Decode Error from Overpass API: {e}")
        print(f"Response status code: {response.status_code if 'response' in locals() else 'N/A'}")
        print(f"Response text: {response.text if 'response' in locals() else 'N/A'}")
        return pd.DataFrame() # Return an empty DataFrame on JSON error
    except RequestException as e:
        print(f"Request Error to Overpass API: {e}")
        return pd.DataFrame() # Return an empty DataFrame on request error

    # Transformando em DataFrame
    pois = []
    for element in data['elements']:
        tags = element.get('tags', {})
        # Determine the 'tipo' based on the presence of specific tags
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

# Agora você pode usar esse df na sua QuadTree!
df_pois = get_rio_pois()
df_pois.head()

# ----- CELL 30 -----
bairros = ['Abolição', 'Acari', 'Água Santa', 'Alto da Boa Vista', 'Anchieta',
       'Andaraí', 'Anil', 'Bancários', 'Bangu', 'Barra da Tijuca',
       'Barra de Guaratiba', 'Barros Filho', 'Benfica', 'Bento Ribeiro',
       'Bonsucesso', 'Botafogo', 'Brás de Pina', 'Cachambi', 'Cacuia',
       'Caju', 'Camorim', 'Campinho', 'Campo dos Afonsos', 'Campo Grande',
       'Cascadura', 'Catete', 'Catumbi', 'Cavalcanti', 'Centro',
       'Cidade de Deus', 'Cidade Nova', 'Cidade Universitária', 'Cocotá',
       'Coelho Neto', 'Colégio', 'Complexo do Alemão', 'Copacabana',
       'Cordovil', 'Cosme Velho', 'Cosmos', 'Costa Barros', 'Curicica',
       'Del Castilho', 'Deodoro', 'Encantado', 'Engenheiro Leal',
       'Engenho da Rainha', 'Engenho de Dentro', 'Engenho Novo',
       'Estácio', 'Flamengo', 'Freguesia (Ilha)',
       'Freguesia (Jacarepaguá)', 'Galeão', 'Gamboa', 'Gardênia Azul',
       'Gávea', 'Gericinó', 'Glória', 'Grajaú', 'Grumari', 'Guadalupe',
       'Guaratiba', 'Higienópolis', 'Honório Gurgel', 'Humaitá',
       'Inhaúma', 'Inhoaíba', 'Ipanema', 'Irajá', 'Itanhangá', 'Jacaré',
       'Jacarepaguá', 'Jacarezinho', 'Jardim América', 'Jardim Botânico',
       'Jardim Carioca', 'Jardim Guanabara', 'Jardim Sulacap', 'Joá',
       'Lagoa', 'Laranjeiras', 'Leblon', 'Leme', 'Lins de Vasconcelos',
       'Madureira', 'Magalhães Bastos', 'Mangueira', 'Manguinhos',
       'Maracanã', 'Maré', 'Marechal Hermes', 'Maria da Graça', 'Méier',
       'Moneró', 'Olaria', 'Osvaldo Cruz', 'Paciência', 'Padre Miguel',
       'Paquetá', 'Parada de Lucas', 'Parque Anchieta', 'Parque Colúmbia',
       'Pavuna', 'Pechincha', 'Pedra de Guaratiba', 'Penha',
       'Penha Circular', 'Piedade', 'Pilares', 'Pitangueiras',
       'Portuguesa', 'Praça da Bandeira', 'Praça Seca',
       'Praia da Bandeira', 'Quintino Bocaiúva', 'Ramos', 'Realengo',
       'Recreio dos Bandeirantes', 'Riachuelo', 'Ribeira',
       'Ricardo de Albuquerque', 'Rio Comprido', 'Rocha', 'Rocha Miranda',
       'Rocinha', 'Sampaio', 'Santa Cruz', 'Santa Teresa', 'Santíssimo',
       'Santo Cristo', 'São Conrado', 'São Cristóvão',
       'São Francisco Xavier', 'Saúde', 'Senador Camará',
       'Senador Vasconcelos', 'Sepetiba', 'Tanque', 'Taquara', 'Tauá',
       'Tijuca', 'Todos os Santos', 'Tomás Coelho', 'Turiaçú', 'Urca',
       'Vargem Grande', 'Vargem Pequena', 'Vasco da Gama', 'Vaz Lobo',
       'Vicente de Carvalho', 'Vidigal', 'Vigário Geral', 'Vila da Penha',
       'Vila Isabel', 'Vila Kosmos', 'Vila Militar', 'Vila Valqueire',
       'Vista Alegre', 'Zumbi']

from unidecode import unidecode

def trata_bairro(b):
  return unidecode(b).strip().lower().replace(' ','-')

bairros_tratados = [trata_bairro(b) for b in bairros]

# ----- CELL 31 -----
import geopandas as gpd
from shapely.geometry import Point

# Handle potential NaN values in latitude/longitude before creating GeoDataFrame
df_anuncios_cleaned = (df_anuncios
                       .dropna(subset=['latitude', 'longitude']))

gdf_anuncios = gpd.GeoDataFrame(
    df_anuncios_cleaned,
    geometry=gpd.points_from_xy(df_anuncios_cleaned.longitude, df_anuncios_cleaned.latitude),
    crs="WGS84"  # WGS84 (padrão para lat/lon)
)

# Reproject gdf_anuncios to match the CRS of df_setores_censitarios
gdf_anuncios = gdf_anuncios.to_crs(df_setores_censitarios.crs)

anuncios_geoloc = gpd.sjoin(gdf_anuncios, df_setores_censitarios[['geometry','cd_bairro','nm_bairro']], how="inner", predicate="within")

anuncios_geoloc.head()

# ----- CELL 32 -----
import geopandas as gpd
from shapely.geometry import Point

# Handle potential NaN values in latitude/longitude before creating GeoDataFrame
df_listings_cleaned = (df_listings
                       .dropna(subset=['latitude', 'longitude']))

gdf_listings = gpd.GeoDataFrame(
    df_listings_cleaned,
    geometry=gpd.points_from_xy(df_listings_cleaned.longitude, df_listings_cleaned.latitude),
    crs="WGS84"  # WGS84 (padrão para lat/lon)
)

# Reproject gdf_anuncios to match the CRS of df_setores_censitarios
gdf_listings = gdf_listings.to_crs(df_setores_censitarios.crs)

listings_geoloc = gpd.sjoin(gdf_listings, df_setores_censitarios[['geometry','cd_bairro','nm_bairro']], how="inner", predicate="within")

listings_geoloc.head()

# ----- CELL 34 -----
import geopandas as gpd
from shapely.geometry import Point

# Create a GeoSeries from geometria_maritima with its original CRS (WGS84)
geometria_maritima_gs = gpd.GeoSeries([geometria_maritima], crs="EPSG:4326")

# Reproject geometria_maritima to the CRS of listings_geoloc for accurate distance calculation
geometria_maritima_reprojected = geometria_maritima_gs.to_crs(listings_geoloc.crs).iloc[0]

# Create Point geometries from latitude and longitude columns of listings_geoloc
listings_points = gpd.points_from_xy(listings_geoloc.longitude, listings_geoloc.latitude)
gdf_listings_points = gpd.GeoSeries(listings_points, crs="EPSG:4326") # Original CRS of lat/lon

# Reproject the listing points to match the CRS of df_setores_censitarios (and geometria_maritima_reprojected)
gdf_listings_points_reprojected = gdf_listings_points.to_crs(listings_geoloc.crs)

# Calculate the minimum distance from each listing (based on lat/lon) to the reprojected maritime geometry
listings_geoloc['distancia_minima_praia'] = gdf_listings_points_reprojected.distance(geometria_maritima_reprojected)

listings_geoloc[['distancia_minima_praia','latitude','longitude','nm_bairro']].head()

# ----- CELL 35 -----
import geopandas as gpd
from shapely.geometry import Point

# Create a GeoSeries from geometria_maritima with its original CRS (WGS84)
geometria_maritima_gs = gpd.GeoSeries([geometria_maritima], crs="EPSG:4326")

# Reproject geometria_maritima to the CRS of anuncios_geoloc for accurate distance calculation
geometria_maritima_reprojected = geometria_maritima_gs.to_crs(anuncios_geoloc.crs).iloc[0]

# Create Point geometries from latitude and longitude columns of anuncios_geoloc
anuncios_points = gpd.points_from_xy(anuncios_geoloc.longitude, anuncios_geoloc.latitude)
gdf_anuncios_points = gpd.GeoSeries(anuncios_points, crs="EPSG:4326") # Original CRS of lat/lon

# Reproject the listing points to match the CRS of df_setores_censitarios (and geometria_maritima_reprojected)
gdf_anuncios_points_reprojected = gdf_anuncios_points.to_crs(anuncios_geoloc.crs)

# Calculate the minimum distance from each listing (based on lat/lon) to the reprojected maritime geometry
anuncios_geoloc['distancia_minima_praia'] = gdf_anuncios_points_reprojected.distance(geometria_maritima_reprojected)

anuncios_geoloc[['distancia_minima_praia','latitude','longitude','nm_bairro']].head()

# ----- CELL 37 -----
tipos_contemplados = ['attraction','subway_entrance', 'school']

# ----- CELL 38 -----
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed
import multiprocessing

def get_nearest_neighbor_info(df_source, df_target, lat_col='latitude', long_col='longitude', n_jobs=-1):
    """
    Retorna um DataFrame com o índice do vizinho mais próximo e a
    distância de Haversine (em km) para cada ponto de df_source.
    """

    # 1. Preparação dos dados em Radianos
    src_rad = np.deg2rad(df_source[[lat_col, long_col]].values)
    tgt_rad = np.deg2rad(df_target[['latitude', 'longitude']].values)

    # 2. Construção da BallTree (Heurística de busca espacial)
    tree = BallTree(tgt_rad, metric='haversine')

    # 3. Função para processamento em paralelo
    def query_chunk(chunk):
        # Retorna distâncias e índices posicionais
        dist, ind = tree.query(chunk, k=1)
        return dist.flatten(), ind.flatten()

    # 4. Divisão em chunks e execução
    num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    chunks = np.array_split(src_rad, num_cores)

    # O zip(*results) separa a lista de tuplas [(d1, i1), (d2, i2)] em duas listas [d1, d2] e [i1, i2]
    results = Parallel(n_jobs=num_cores)(
        delayed(query_chunk)(chunk) for chunk in chunks
    )
    distances_rad, positions = zip(*results)

    # 5. Consolidação e Mapeamento
    # Concatenamos os arrays e convertemos radianos para KM (Raio da Terra ~6371)
    final_distances = np.concatenate(distances_rad) * 6371
    final_positions = np.concatenate(positions)

    # Criamos o DataFrame de saída respeitando o índice original do df_source
    return pd.DataFrame({
        'nearest_index': df_target.index[final_positions],
        'distance_km': final_distances
    }, index=df_source.index)

# --- Exemplo de Uso prático para Imóveis ---
# result_df = get_nearest_neighbor_info(df_imoveis, df_estacoes_metro)
# df_imoveis = df_imoveis.join(result_df)

# ----- CELL 39 -----
for t in tipos_contemplados:
  df_target = df_pois[df_pois['tipo']==t].reset_index(drop=True)
  nearest_info = get_nearest_neighbor_info(anuncios_geoloc, df_target)
  nearest_info[f'nome_{t}_mais_proximo'] = df_target.iloc[nearest_info['nearest_index'].values]['nome'].values
  anuncios_geoloc[[f'distancia_minima_{t}',f'nome_{t}_mais_proximo']] = nearest_info[['distance_km',f'nome_{t}_mais_proximo']].values
  anuncios_geoloc[f'distancia_minima_{t}'] = anuncios_geoloc[f'distancia_minima_{t}'].astype(float)

# ----- CELL 40 -----
for t in tipos_contemplados:
  df_target = df_pois[df_pois['tipo']==t].reset_index(drop=True)
  nearest_info = get_nearest_neighbor_info(listings_geoloc, df_target)
  nearest_info[f'nome_{t}_mais_proximo'] = df_target.iloc[nearest_info['nearest_index'].values]['nome'].values
  listings_geoloc[[f'distancia_minima_{t}',f'nome_{t}_mais_proximo']] = nearest_info[['distance_km',f'nome_{t}_mais_proximo']].values
  listings_geoloc[f'distancia_minima_{t}'] = listings_geoloc[f'distancia_minima_{t}'].astype(float)

# ----- CELL 41 -----
anuncios_geoloc.head()

# ----- CELL 42 -----
listings_geoloc.head()

# ----- CELL 44 -----
anuncios_por_bairro = anuncios_geoloc.groupby('nm_bairro').size().reset_index(name='Qtde de anúncios no quinto andar')
listings_por_bairro = listings_geoloc.groupby('nm_bairro').size().reset_index(name='Qtde de anúncios no airbnb')
contagem = pd.merge(anuncios_por_bairro, listings_por_bairro, on='nm_bairro', how='inner')
contagem['% de anúncios no quinto andar'] = contagem['Qtde de anúncios no quinto andar']/contagem['Qtde de anúncios no quinto andar'].sum()
contagem['% de anúncios no airbnb'] = contagem['Qtde de anúncios no airbnb']/contagem['Qtde de anúncios no airbnb'].sum()
contagem['Média harmonica das %'] = 2/(1/contagem['% de anúncios no quinto andar'] + 1/contagem['% de anúncios no airbnb'])

# ----- CELL 45 -----
contagem = contagem.sort_values(by='Média harmonica das %',ascending=False).reset_index(drop=True)

# ----- CELL 46 -----
contagem['% acumulada de anúncios no quinto andar'] = contagem['% de anúncios no quinto andar'].cumsum()
contagem['% acumulada de anúncios no airbnb'] = contagem['% de anúncios no airbnb'].cumsum()
contagem['Média das % acumuladas'] = (contagem['% acumulada de anúncios no quinto andar'] + contagem['% acumulada de anúncios no airbnb'])/2

# ----- CELL 47 -----
plt.figure(figsize=(12, 6))
sns.lineplot(contagem[['% de anúncios no quinto andar','% de anúncios no airbnb','Média harmonica das %']])
plt.title('Distribuição da Porcentagem de Anúncios por Bairro')
plt.xlabel('Número do Bairro (Ordenado por Média Harmônica)')
plt.ylabel('Porcentagem de Anúncios')
plt.show()

# ----- CELL 48 -----
contagem = contagem.sort_values(by='Média das % acumuladas',ascending=True).reset_index(drop=True)

# ----- CELL 49 -----
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plotting cumulative percentages on the primary y-axis
sns.lineplot(x=contagem.index, y='% acumulada de anúncios no quinto andar', data=contagem, ax=ax1, label='% Acumulada Quinto Andar', color='blue')
sns.lineplot(x=contagem.index, y='% acumulada de anúncios no airbnb', data=contagem, ax=ax1, label='% Acumulada Airbnb', color='green')
sns.lineplot(x=contagem.index, y='Média das % acumuladas', data=contagem, ax=ax1, label='Média das % Acumuladas', color='red', linestyle='--')

# Add horizontal line at 80% cumulative average
ax1.axhline(y=0.8, color='orange', linestyle=':', label='80% Média Acumulada')

ax1.set_xlabel('Número do Bairro (Ordenado por Média Acumulada)')
ax1.set_ylabel('Porcentagem Acumulada de Anúncios', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend(loc='upper left')

# Create a secondary y-axis
ax2 = ax1.twinx()

# Plotting quantities on the secondary y-axis
sns.lineplot(x=contagem.index, y='Qtde de anúncios no quinto andar', data=contagem, ax=ax2, label='Qtde Quinto Andar', color='cyan', linestyle=':')
sns.lineplot(x=contagem.index, y='Qtde de anúncios no airbnb', data=contagem, ax=ax2, label='Qtde Airbnb', color='lime', linestyle=':')

ax2.set_ylabel('Quantidade de Anúncios', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')
ax2.legend(loc='upper right')

plt.title('Porcentagem Acumulada e Quantidade de Anúncios por Bairro')
plt.tight_layout()
plt.show()

# ----- CELL 50 -----
pct_corte = 0.8
filter_mask = contagem[contagem['Média das % acumuladas']>=.8]
bairros_contemplados = filter_mask['nm_bairro'].tolist()
idx_corte = filter_mask.index[0]

print(f'O indice de corte é {idx_corte}')
print(f'Sobraram {filter_mask.size} bairros')
print(f'Bairros contemplados:')
for i in bairros_contemplados:
  print(f'-{i}')

# ----- CELL 51 -----
# anuncios_geoloc = anuncios_geoloc[anuncios_geoloc['nm_bairro'].isin(bairros_contemplados)].reset_index(drop=True)
# listings_geoloc = listings_geoloc[listings_geoloc['nm_bairro'].isin(bairros_contemplados)].reset_index(drop=True)

# ----- CELL 53 -----
def plot_singleVar_numeric(df, col, xlim=None, ylim=None):
  plt.figure(figsize=(18, 6))

  # Histogram for média_valor_transação
  plt.subplot(1, 2, 1)
  sns.histplot(df[col].dropna(), kde=True, bins=50)
  plt.title(f'Distribuição: {col}')
  plt.xlabel(col)
  plt.ylabel('Frequencia')

  # Box plot for média_valor_transação
  plt.subplot(1, 2, 2)
  sns.boxplot(y=df[col].dropna())
  plt.title(f'BoxPlot: {col}')
  plt.ylabel(col)
  if xlim:
    plt.xlim(xlim)
  if ylim:
    plt.ylim(ylim)

  plt.tight_layout()
  plt.show()

# ----- CELL 54 -----
def plot_singleVar_categorical(df, col):
  descending_order = df[col].value_counts().index

  plt.figure(figsize=(18, 6))
  ax = sns.countplot(x=col, data=df, palette='viridis', hue=col, legend=False, order=descending_order)
  plt.title(f'Distribuição: {col}')
  plt.xlabel(col)
  plt.ylabel('Contagem')
  plt.xticks(rotation=45, ha='right')

  # Add value labels on top of each bar
  for p in ax.patches:
      ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                  textcoords='offset points')

  plt.tight_layout()
  plt.show()

# ----- CELL 55 -----
def plot_against_target_categorical(df, col, target):
  descending_order = df.groupby(col)[target].median().sort_values().index

  plt.figure(figsize=(18, 6))
  sns.boxplot(x=col, y=target, data=df, palette='viridis', hue=col, legend=False, order=descending_order)
  plt.title(f'{target} por {col}')
  plt.xlabel(col)
  plt.ylabel(target)
  plt.xticks(rotation=45, ha='right')
  plt.tight_layout()
  plt.show()

# ----- CELL 56 -----
def plot_against_target_numerical(df, col, target):
  from scipy.stats import pearsonr # Import pearsonr

  plt.figure(figsize=(18, 6))
  sns.scatterplot(x=col, y=target, data=df, alpha=0.6)

  # Calculate correlation and p-value, handling potential NaN values
  df_cleaned = df[[col, target]].dropna()
  if not df_cleaned.empty:
      correlation, p_value = pearsonr(df_cleaned[col], df_cleaned[target])
      annotation_text = f'Correlação (Pearson): {correlation:.2f}\nP-valor: {p_value:.3f}'
      plt.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.9))
  else:
      plt.annotate('Não foi possível calcular a correlação (dados insuficientes ou NaN)', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.5))

  plt.title(f'{target} vs {col}')
  plt.xlabel(col)
  plt.ylabel(target)
  plt.tight_layout()
  plt.show()

# ----- CELL 58 -----
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

class Point:
    def __init__(self, x, y, value):
        self.x, self.y, self.value = x, y, value

class Boundary:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def contains(self, point):
        return (self.x - self.w <= point.x <= self.x + self.w and
                self.y - self.h <= point.y <= self.y + self.h)

    def get_polygon(self):
        """Retorna a geometria do Boundary como um Polygon."""
        min_x = self.x - self.w
        max_x = self.x + self.w
        min_y = self.y - self.h
        max_y = self.y + self.h
        return Polygon([
            (min_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y),
            (min_x, min_y)
        ])

class QuadTree:
    def __init__(self, boundary, capacity, depth=0, uid="0"):
        self.boundary = boundary
        self.capacity = capacity
        self.points = []
        self.divided = False
        self.depth = depth
        self.uid = uid # Identificador único do quadrante

    def subdivide(self):
        x, y, w, h = self.boundary.x, self.boundary.y, self.boundary.w, self.boundary.h

        # Criando sub-quadrantes com IDs hierárquicos (ex: 0.1, 0.2...)
        self.northwest = QuadTree(Boundary(x - w/2, y + h/2, w/2, h/2), self.capacity, self.depth + 1, f"{self.uid}.1")
        self.northeast = QuadTree(Boundary(x + w/2, y + h/2, w/2, h/2), self.capacity, self.depth + 1, f"{self.uid}.2")
        self.southwest = QuadTree(Boundary(x - w/2, y - h/2, w/2, h/2), self.capacity, self.depth + 1, f"{self.uid}.3")
        self.southeast = QuadTree(Boundary(x + w/2, y - h/2, w/2, h/2), self.capacity, self.depth + 1, f"{self.uid}.4")
        self.divided = True

    def insert(self, point):
        if not self.boundary.contains(point):
            return False

        if len(self.points) < self.capacity and not self.divided:
            self.points.append(point)
            return True
        else:
            if not self.divided:
                self.subdivide()
                # Move os pontos existentes para os novos quadrantes
                old_points = self.points
                self.points = []
                for p in old_points:
                    self._insert_to_children(p)

            return self._insert_to_children(point)

    def _insert_to_children(self, point):
        return (self.northwest.insert(point) or self.northeast.insert(point) or
                self.southwest.insert(point) or self.southeast.insert(point))

    def get_quadrant_id(self, point):
        """Retorna o UID do menor quadrante (folha) que contém o ponto."""
        if not self.boundary.contains(point):
            return None

        if not self.divided:
            return self.uid

        # Busca recursiva nos filhos
        for child in [self.northwest, self.northeast, self.southwest, self.southeast]:
            res = child.get_quadrant_id(point)
            if res: return res
        return self.uid

    def get_quadrant_geometry(self, quad_uid):
        """Retorna a geometria (Polygon) de um quadrante dado o seu UID."""
        if self.uid == quad_uid:
            return self.boundary.get_polygon()

        if not self.divided:
            return None

        for child in [self.northwest, self.northeast, self.southwest, self.southeast]:
            geometry = child.get_quadrant_geometry(quad_uid)
            if geometry:
                return geometry
        return None

# --- Configuração RJ ---
lat_rj, lon_rj = (-22.7037, -23.1856), (-43.0846, -43.8086)

def get_params(lat_t, lon_t):
    return sum(lon_t)/2, sum(lat_t)/2, abs(lon_t[0]-lon_t[1])/2, abs(lat_t[0]-lat_t[1])/2

c_lon, c_lat, w, h = get_params(lat_rj, lon_rj)


# ----- CELL 61 -----
anuncios_geoloc.info()

# ----- CELL 62 -----
rename_anuncios = {
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    'hasFurniture': 'Mobiliado',
    'salePrice': 'Valor de venda',
    'area': 'Área construída',
    'bathrooms': 'Banheiros',
    'bedrooms': 'Quartos',
    'parkingSpaces': 'Vagas',
    'nm_bairro': 'Bairro',
    'street': 'Logradouro',
    'condoPrice': 'Valor do condominio',
    'distancia_minima_praia': 'Distancia da praia',
    'distancia_minima_attraction': 'Distância minima de atracao turistica',
    'distancia_minima_subway_entrance': 'Distância minima de entrada de metrô',
    'distancia_minima_school': 'Distância minima de escola'
    # 'nome_attraction_mais_proximo': 'Atracao turistica mais proxima'
}

target_anuncios = 'Valor de venda'
num_features_anuncios = [x[1] for x in rename_anuncios.items() if (anuncios_geoloc[x[0]].dtype in (int,float)) and (x[1] != target_anuncios)]
cat_features_anuncios = [x[1] for x in rename_anuncios.items() if anuncios_geoloc[x[0]].dtype in (object,bool)]
# for i in rename_anuncios.values():
#   print(i)

# ----- CELL 63 -----
num_features_anuncios

# ----- CELL 64 -----
cat_features_anuncios

# ----- CELL 65 -----
anuncios_geoloc.rename(columns=rename_anuncios,inplace=True)

# ----- CELL 66 -----
print(f"Shape of df_anuncios: {anuncios_geoloc.shape}")
anuncios_geoloc.describe()

# ----- CELL 69 -----
missing_values = anuncios_geoloc.isnull().sum()
missing_percentage = (missing_values / len(anuncios_geoloc)) * 100
missing_df = pd.DataFrame({
    'Column': missing_percentage.index,
    'Percentage': missing_percentage.values
})
missing_df = missing_df[missing_df['Percentage'] > 0].sort_values(by='Percentage', ascending=False)

print("Missing Values Percentage in anuncios_geoloc:")
print(missing_df)

# ----- CELL 71 -----
# if not missing_df.empty:
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x='Column', y='Percentage', data=missing_df, palette='viridis')
#     plt.title('Percentage of Missing Values per Column in df_transacoes')
#     plt.xlabel('Columns')
#     plt.ylabel('Percentage of Missing Values')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()
# else:
#     print("No missing values to display.")

# ----- CELL 74 -----
plot_singleVar_numeric(anuncios_geoloc,'Valor de venda')

# ----- CELL 75 -----
anuncios_geoloc['Valor de venda (escala log)'] = anuncios_geoloc['Valor de venda'].map(np.log1p)
targetlog_anuncios = 'Valor de venda (escala log)'

# ----- CELL 76 -----
plot_singleVar_numeric(anuncios_geoloc,'Valor de venda (escala log)')

# ----- CELL 78 -----
for var in cat_features_anuncios:
  if var not in ['Logradouro']:
    plot_singleVar_categorical(anuncios_geoloc, var)
    print()
    plot_against_target_categorical(anuncios_geoloc, var, 'Valor de venda (escala log)')
    # plot_against_target_categorical(anuncios_geoloc, var, 'Valor de venda')
    print()

# ----- CELL 80 -----
for var in num_features_anuncios:
  if var not in ['Latitude', 'Longitude']:
    if var in ['Valor do condominio','Área construída']:
      anuncios_geoloc[f'{var} (escala log)'] = anuncios_geoloc[var].map(np.log1p)
      var = f'{var} (escala log)'
    plot_singleVar_numeric(anuncios_geoloc, var)
    print()
    plot_against_target_numerical(anuncios_geoloc, var, 'Valor de venda (escala log)')
    print()

# ----- CELL 82 -----
bairro_avg_value = anuncios_geoloc.groupby('Bairro')[target_anuncios].mean().sort_values(ascending=False).head(10).reset_index()

plt.figure(figsize=(14, 8))
ax = sns.barplot(x=target_anuncios, y='Bairro', data=bairro_avg_value, palette='viridis', hue='Bairro', legend=False)
plt.title('Top 10 Bairros por Valor Transação')
plt.xlabel('Valor Transação')
plt.ylabel('Bairro')

# Add value labels on top of each bar
for p in ax.patches:
    width = p.get_width()
    plt.text(width, p.get_y() + p.get_height() / 2, f'{width:,.0f}', va='center')

plt.xlim(0, bairro_avg_value[target_anuncios].max() * 1.1) # Increase x-axis limit by 15%

plt.tight_layout()
plt.show()

# ----- CELL 83 -----
bairro_avg_value_for_merge = anuncios_geoloc.groupby('Bairro')[target_anuncios].mean().reset_index()

df_bairros_polygons = df_setores_censitarios.groupby('nm_bairro')['geometry'].agg(lambda x: x.union_all()).reset_index()
df_bairros_polygons = gpd.GeoDataFrame(df_bairros_polygons, geometry='geometry', crs=df_setores_censitarios.crs)
df_bairros_polygons.rename(columns={'nm_bairro': 'Bairro'}, inplace=True)

df_plot_bairros = df_bairros_polygons.merge(bairro_avg_value_for_merge, on='Bairro', how='left')

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
df_plot_bairros.dropna(subset=[target_anuncios]).plot(column=target_anuncios, cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True,
                     legend_kwds={'label': f"Valor Médio do {target_anuncios}", 'orientation': "horizontal"})

ax.set_title(f'Valor Médio do {target_anuncios} por Bairro (Polígonos)')
ax.set_axis_off()
plt.show()

# ----- CELL 85 -----
corr_cols_anuncios = ['Valor de venda (escala log)'] + [x for x in num_features_anuncios if x not in ['Latitude','Longitude']]
plt.figure(figsize=(12, 10))
sns.heatmap(anuncios_geoloc[corr_cols_anuncios].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlações das Features na base de vendas')
plt.tight_layout()
plt.show()

# ----- CELL 87 -----
import sys
sys.setrecursionlimit(100000) # Increased recursion limit further

# # Criando DataFrame
# df = pd.DataFrame({
#     'latitude': np.random.uniform(min(lat_rj), max(lat_rj), 100),
#     'longitude': np.random.uniform(min(lon_rj), max(lon_rj), 100),
#     'valor': np.random.randint(0, 100, 100)
# })

# Construindo a árvore
qt_anuncios = QuadTree(Boundary(c_lon, c_lat, w, h), capacity=500)
for _, row in anuncios_geoloc.iterrows():
    qt_anuncios.insert(Point(row['Longitude'], row['Latitude'], row[targetlog_anuncios]))

# --- ATRIBUIÇÃO AO DATAFRAME ---
# Criamos a nova coluna usando uma função lambda que consulta a árvore
anuncios_geoloc['quadrante'] = anuncios_geoloc.apply(
    lambda row: qt_anuncios.get_quadrant_id(Point(row['Longitude'], row['Latitude'], None)),
    axis=1
)
anuncios_geoloc['poligono_quadrante'] = anuncios_geoloc.apply(
    lambda row: qt_anuncios.get_quadrant_geometry(row['quadrante']),
    axis=1
)

anuncios_geoloc.head(10)

# ----- CELL 88 -----
import matplotlib.pyplot as plt
import geopandas as gpd

# Calculate the mean 'Valor de venda' for each quadrant
quadrant_avg_value = anuncios_geoloc.groupby('quadrante')['Valor de venda'].mean().reset_index()

# Get unique quadrant polygons and create a GeoDataFrame
quadrant_polygons = anuncios_geoloc[['quadrante', 'poligono_quadrante']].drop_duplicates(subset=['quadrante'])

gdf_quadrants = gpd.GeoDataFrame(quadrant_polygons,
                                 geometry='poligono_quadrante',
                                 crs=anuncios_geoloc.crs)

# Merge the average value with the quadrant GeoDataFrame
gdf_plot_quadrants = gdf_quadrants.merge(quadrant_avg_value, on='quadrante', how='left')

# Plot the map
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gdf_plot_quadrants.dropna(subset=['Valor de venda']).plot(column='Valor de venda', cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True,
                     legend_kwds={'label': "Valor Médio de Venda por Quadrante", 'orientation': "horizontal"})

ax.set_title('Valor Médio de Venda por Quadrante (QuadTree)')
ax.set_axis_off()
plt.show()

# ----- CELL 90 -----
print(f"Shape of df_listings: {listings_geoloc.shape}")
listings_geoloc.info()
listings_geoloc.describe(include='all')

# ----- CELL 91 -----
listings_geoloc.head()

# ----- CELL 92 -----
rename_listings = {
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    'room_type': 'Tipo de locação',
    'nm_bairro':'Bairro',
    'price': 'Preço do aluguel',
    'bathrooms':'Banheiros',
    'bedrooms': 'Quartos',
    'beds': 'Camas',
    'accommodates': 'Qtd de pessoas acomodadas',
    'minimum_nights': 'Mínimo de noites',
    'number_of_reviews': 'Número de reviews',
    'reviews_per_month': 'Reviews por mês',
    'calculated_host_listings_count': 'Contagem de hosts',
    'availability_365': 'Disponibilidade no ano',
    'distancia_minima_praia': 'Distancia da praia',
    'distancia_minima_attraction': 'Distância minima de atracao turistica',
    'distancia_minima_subway_entrance': 'Distância minima de entrada de metrô',
    'distancia_minima_school': 'Distância minima de escola',
    'estimated_days_rented_ltm': 'Dias alugados (estimado)'
    # 'nome_attraction_mais_proximo': 'Atracao turistica mais proxima'
}


target_listings = 'Preço do aluguel'
num_features_listings = [x[1] for x in rename_listings.items() if (listings_geoloc[x[0]].dtype in (int,float)) and (x[1] != target_listings)]
cat_features_anuncios = [x[1] for x in rename_listings.items() if listings_geoloc[x[0]].dtype in (object,bool)]
# for i in rename_listings.values():
#   print(i)

# ----- CELL 93 -----
listings_geoloc.rename(columns=rename_listings, inplace=True)

# ----- CELL 95 -----
plot_singleVar_numeric(listings_geoloc,target_listings)

# ----- CELL 96 -----
listings_geoloc[f'{target_listings} (escala log)'] = listings_geoloc[target_listings].map(np.log1p)
plot_singleVar_numeric(listings_geoloc,f'{target_listings} (escala log)')

# ----- CELL 97 -----
targetlog_listings = f'{target_listings} (escala log)'

# ----- CELL 99 -----
for var in cat_features_anuncios:
  plot_singleVar_categorical(listings_geoloc, var)
  print()
  plot_against_target_categorical(listings_geoloc, var, targetlog_listings)
  print()

# ----- CELL 101 -----
for var in num_features_listings:
  if var not in ['Latitude', 'Longitude']:
    plot_singleVar_numeric(listings_geoloc, var)
    print()
    plot_against_target_numerical(listings_geoloc, var, targetlog_listings)
    print()

# ----- CELL 103 -----
bairro_avg_value = listings_geoloc.groupby('Bairro')[target_listings].mean().sort_values(ascending=False).head(10).reset_index()

plt.figure(figsize=(14, 8))
ax = sns.barplot(x=target_listings, y='Bairro', data=bairro_avg_value, palette='viridis', hue='Bairro', legend=False)
plt.title('Top 10 Bairros por Valor Transação')
plt.xlabel('Valor Transação')
plt.ylabel('Bairro')

# Add value labels on top of each bar
for p in ax.patches:
    width = p.get_width()
    plt.text(width, p.get_y() + p.get_height() / 2, f'{width:,.0f}', va='center')

plt.xlim(0, bairro_avg_value[target_listings].max() * 1.1) # Increase x-axis limit by 15%

plt.tight_layout()
plt.show()

# ----- CELL 104 -----
bairro_avg_value_for_merge = listings_geoloc.groupby('Bairro')[target_listings].mean().reset_index()

df_bairros_polygons = df_setores_censitarios.groupby('nm_bairro')['geometry'].agg(lambda x: x.union_all()).reset_index()
df_bairros_polygons = gpd.GeoDataFrame(df_bairros_polygons, geometry='geometry', crs=df_setores_censitarios.crs)
df_bairros_polygons.rename(columns={'nm_bairro': 'Bairro'}, inplace=True)

df_plot_bairros = df_bairros_polygons.merge(bairro_avg_value_for_merge, on='Bairro', how='left')

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
df_plot_bairros.dropna(subset=[target_listings]).plot(column=target_listings, cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True,
                     legend_kwds={'label': f"Valor Médio do {target_listings}", 'orientation': "horizontal"})

ax.set_title(f'Valor Médio do {target_listings} por Bairro (Polígonos)')
ax.set_axis_off()
plt.show()

# ----- CELL 106 -----
# corr_cols_listings

# ----- CELL 107 -----
corr_cols_listings = [targetlog_listings] + [x for x in num_features_listings if x not in ['Latitude','Longitude']]

vars_de_interesse = ['Preço do aluguel (escala log)',
                     'Banheiros','Quartos','Qtd de pessoas acomodadas',
                     'Mínimo de noites','Reviews por mês','Disponibilidade no ano',
 'Distancia da praia',
 'Distância minima de atracao turistica',
 'Distância minima de entrada de metrô',
 'Distância minima de escola']

plt.figure(figsize=(12, 10))
sns.heatmap(listings_geoloc[vars_de_interesse].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlações das Features na base de vendas')
plt.tight_layout()
plt.show()

# ----- CELL 109 -----
import sys
sys.setrecursionlimit(100000) # Increased recursion limit further

# # Criando DataFrame
# df = pd.DataFrame({
#     'latitude': np.random.uniform(min(lat_rj), max(lat_rj), 100),
#     'longitude': np.random.uniform(min(lon_rj), max(lon_rj), 100),
#     'valor': np.random.randint(0, 100, 100)
# })

# Construindo a árvore
qt_listings = QuadTree(Boundary(c_lon, c_lat, w, h), capacity=100)
for _, row in listings_geoloc.iterrows():
    qt_listings.insert(Point(row['Longitude'], row['Latitude'], row[targetlog_listings]))

# --- ATRIBUIÇÃO AO DATAFRAME ---
# Criamos a nova coluna usando uma função lambda que consulta a árvore
listings_geoloc['quadrante'] = listings_geoloc.apply(
    lambda row: qt_listings.get_quadrant_id(Point(row['Longitude'], row['Latitude'], None)),
    axis=1
)
listings_geoloc['poligono_quadrante'] = listings_geoloc.apply(
    lambda row: qt_listings.get_quadrant_geometry(row['quadrante']),
    axis=1
)

listings_geoloc.head(10)

# ----- CELL 110 -----
import matplotlib.pyplot as plt
import geopandas as gpd

# Calculate the mean 'Valor de venda' for each quadrant
quadrant_avg_value = listings_geoloc.groupby('quadrante')[target_listings].mean().reset_index()

# Get unique quadrant polygons and create a GeoDataFrame
quadrant_polygons = listings_geoloc[['quadrante', 'poligono_quadrante']].drop_duplicates(subset=['quadrante'])

gdf_quadrants = gpd.GeoDataFrame(quadrant_polygons,
                                 geometry='poligono_quadrante',
                                 crs=listings_geoloc.crs)

# Merge the average value with the quadrant GeoDataFrame
gdf_plot_quadrants = gdf_quadrants.merge(quadrant_avg_value, on='quadrante', how='left')

# Plot the map
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gdf_plot_quadrants.dropna(subset=[target_listings]).plot(column=target_listings, cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True,
                     legend_kwds={'label': f"Valor Médio de Aluguel por Quadrante", 'orientation': "horizontal"})

ax.set_title('Valor Médio de Aluguel por Quadrante (QuadTree)')
ax.set_axis_off()
plt.show()

# ----- CELL 113 -----
anuncios_geoloc.head(5)

# ----- CELL 114 -----
anuncios_geoloc.to_csv(path_processed + 'base_features_quintoAndar.csv')

# ----- CELL 116 -----
listings_geoloc.head(5)

# ----- CELL 117 -----
listings_geoloc.to_csv(path_processed + 'base_features_AirBnb.csv')

# ----- CELL 118 -----


# ----- CELL 119 -----


