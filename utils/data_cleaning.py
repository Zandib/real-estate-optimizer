import pandas as pd
from unidecode import unidecode

def trata_valor(x):
    """
    Remove caracteres indesejados (como $ e ,) e converte para float.
    """
    x = str(x).replace('$','').replace(',','')
    return float(x)

def trata_bairro(b):
    """
    Padroniza os nomes de bairros retirando acentos e espaços.
    """
    return unidecode(b).strip().lower().replace(' ','-')

def estimate_rented_days(df):
    """
    Estima a quantidade de dias alugados baseado na quantidade de reviews.
    """
    REVIEW_RATE = 0.5  # Assume-se que 50% dos hóspedes deixam reviews
    DEFAULT_STAY = 3   # Média de noites caso o minimum_nights seja irreal

    df_calc = df.copy()

    # Tratamento do minimum_nights
    df_calc['clamped_min_nights'] = df_calc['minimum_nights'].clip(upper=30)

    # Estimativa de Reservas Totais (Total Bookings)
    df_calc['estimated_bookings'] = df_calc['number_of_reviews'] / REVIEW_RATE

    # Cálculo da Coluna Alvo: Dias Alugados (Estimado)
    df_calc['estimated_days_rented'] = df_calc['estimated_bookings'] * df_calc['clamped_min_nights']

    # Estimativa Anual
    df_calc['estimated_days_rented_ltm'] = (df_calc['number_of_reviews_ltm'] / REVIEW_RATE) * df_calc['clamped_min_nights']
    df_calc['estimated_days_rented_ltm'] = df_calc['estimated_days_rented_ltm'].clip(upper=365)

    return df_calc
