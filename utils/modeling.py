"""
utils/modeling.py
-----------------
Módulo com classes e funções auxiliares para o pipeline de modelagem preditiva.
Inclui transformadores sklearn-compatible e funções de feature engineering.

Conteúdo:
    - MeanMapper: Target mean-encoding para features categóricas.
    - create_target_derived_features: Features de diferença de preço vs. média do bairro/quadrante.
    - CustomPreprocessingPipeline: Pipeline completo de pré-processamento compatível com sklearn.
      IMPORTANTE: Este pipeline é genérico (sem VIF/p-value). A seleção de variáveis via VIF e
      p-valor é exclusiva do modelo OLS e deve ser feita externamente (ver run_training.py).
    - calculate_vif: Calcula VIF para todas as colunas de um DataFrame.
    - iterative_vif_removal: Remove variáveis com VIF acima do threshold (exclusivo OLS).
    - iterative_pvalue_removal: Backward selection por p-valor (exclusivo OLS).
    - save_model: Serializa qualquer objeto com cloudpickle.
"""

import numpy as np
import pandas as pd
import cloudpickle as pickle
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ==============================================================================
# 1. MEAN MAPPER — Target Mean Encoding para Variáveis Categóricas
# ==============================================================================

class MeanMapper(BaseEstimator, TransformerMixin):
    """
    Realiza target mean encoding em features categóricas.

    Para cada categoria, calcula e armazena:
        - a média do target (mean)
        - o desvio padrão do target (std)
        - a contagem de observações (count)

    Em .transform(), substitui cada valor categórico pelos três
    valores estatísticos calculados no .fit(), gerando 3 colunas
    numéricas por feature categórica original.

    Usado por: OLS, XGBoost, KNN (todos os modelos).
    """

    def __init__(self):
        self.mapping_mean = {}
        self.mapping_std = {}
        self.mapping_count = {}
        self.failsafe_mean = 0
        self.failsafe_std = 0
        self.failsafe_count = -1  # -1 sinaliza categoria nunca vista no treino

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Aprende as estatísticas (mean, std, count) do target para cada
        valor único de cada feature categórica no conjunto de treino.

        Args:
            X (pd.DataFrame): Features categóricas.
            y (pd.Series): Variável alvo (target).
        """
        df = X.copy()
        df['target'] = y.values

        for feature in X.columns:
            grouped = df.groupby(feature, observed=False)['target']
            self.mapping_mean[feature] = grouped.mean()
            self.mapping_std[feature] = grouped.std()
            self.mapping_count[feature] = grouped.count()

        self.failsafe_mean = y.mean()
        self.failsafe_std = y.std()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Substitui as colunas categóricas pelas estatísticas aprendidas no fit.
        Categorias não vistas no treino recebem os valores de failsafe.

        Args:
            X (pd.DataFrame): Features categóricas.

        Returns:
            pd.DataFrame: DataFrame com 3 colunas numéricas por feature original
                          ({feature}_mean, {feature}_std, {feature}_count).
        """
        df = X[list(self.mapping_mean.keys())].copy()

        for feature in self.mapping_mean.keys():
            df[feature + '_mean'] = (
                df[feature]
                .map(self.mapping_mean[feature])
                .astype(float)
                .fillna(self.failsafe_mean)
            )
            df[feature + '_std'] = (
                df[feature]
                .map(self.mapping_std[feature])
                .astype(float)
                .fillna(self.failsafe_std)
            )
            df[feature + '_count'] = (
                df[feature]
                .map(self.mapping_count[feature])
                .astype(float)
                .fillna(self.failsafe_count)
            )

        # Remove as colunas categóricas originais
        df = df.drop(columns=list(self.mapping_mean.keys()))
        return df


# ==============================================================================
# 2. CREATE_TARGET_DERIVED_FEATURES — Feature Engineering de Diferença de Preço
# ==============================================================================

def create_target_derived_features(
    X_data: pd.DataFrame,
    y_data: pd.Series,
    bairro_mean_col: str = 'Bairro_mean',
    quadrante_mean_col: str = 'quadrante_mean'
) -> pd.DataFrame:
    """
    Cria features que capturam o quanto o preço de um imóvel desvia da média
    do seu bairro e do seu quadrante espacial.

    Usada exclusivamente no pipeline de modelagem de DISPONIBILIDADE (dias alugados),
    pois adiciona contexto de preço relativo ao modelo de ocupação.

    IMPORTANTE: Esta função usa o target real (y_data) para calcular as diferenças.
    Para evitar data leakage, deve ser chamada separadamente para treino e teste,
    usando o y_test real apenas para avaliação (o que é aceitável neste contexto pois
    não é usada para ajuste do modelo de disponibilidade em si).

    Args:
        X_data (pd.DataFrame): DataFrame com as features já mean-encoded.
                               Deve conter 'Bairro_mean' e 'quadrante_mean'.
        y_data (pd.Series): Variável alvo de preço (escala log).
        bairro_mean_col (str): Nome da coluna de média do bairro em X_data.
        quadrante_mean_col (str): Nome da coluna de média do quadrante em X_data.

    Returns:
        pd.DataFrame: DataFrame com as colunas:
                      - 'Diferenca_preco_bairro'
                      - 'Diferenca_preco_quadrante'
    """
    new_features = pd.DataFrame(index=X_data.index)
    aligned_y = y_data.loc[X_data.index]

    if bairro_mean_col in X_data.columns:
        new_features['Diferenca_preco_bairro'] = aligned_y - X_data[bairro_mean_col]
    else:
        print(f"[AVISO] Coluna '{bairro_mean_col}' não encontrada. Preenchendo com NaN.")
        new_features['Diferenca_preco_bairro'] = np.nan

    if quadrante_mean_col in X_data.columns:
        new_features['Diferenca_preco_quadrante'] = aligned_y - X_data[quadrante_mean_col]
    else:
        print(f"[AVISO] Coluna '{quadrante_mean_col}' não encontrada. Preenchendo com NaN.")
        new_features['Diferenca_preco_quadrante'] = np.nan

    return new_features


# ==============================================================================
# 3. CUSTOM PREPROCESSING PIPELINE — Pipeline Base (sem seleção de variáveis)
# ==============================================================================

class CustomPreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    Pipeline de pré-processamento sklearn-compatible que encapsula:
        1. Imputação de valores faltantes (fillna com médias de treino)
        2. Target Mean Encoding das features categóricas (via MeanMapper)
        3. Normalização numérica (StandardScaler)

    IMPORTANTE sobre separação de pipelines por modelo:
    -------------------------------------------------------
    Este pipeline NÃO inclui remoção de variáveis por VIF ou p-valor.
    Essas etapas são específicas do modelo OLS/SAR e são feitas de forma
    iterativa no script run_training.py, antes do ajuste do modelo linear.

    XGBoost e KNN usam este pipeline diretamente (sem seleção de variáveis),
    pois são métodos não-paramétricos que toleram multicolinearidade.

    Parâmetros do construtor:
        initial_features (list): Lista original de features (antes de qualquer transformação).
        initial_cat_features (list): Subconjunto de initial_features que são categóricas.
        fillna_values (dict): Dicionário {coluna: valor_de_preenchimento} para imputação.
    """

    def __init__(
        self,
        initial_features: list,
        initial_cat_features: list,
        fillna_values: dict
    ):
        self.initial_features = initial_features
        self.initial_cat_features = initial_cat_features
        self.fillna_values = fillna_values

        self.mean_mapper = MeanMapper()
        self.scaler = StandardScaler()
        self.final_features_order = None  # Ordem exata das colunas após transformação

    def fit(self, X_raw: pd.DataFrame, y: pd.Series):
        """
        Aprende: médias do MeanMapper e parâmetros do StandardScaler
        com base exclusivamente no conjunto de treino.

        Args:
            X_raw (pd.DataFrame): Dados brutos de treino (com as initial_features).
            y (pd.Series): Target de treino (para o MeanMapper).
        """
        X = X_raw[self.initial_features].copy()
        X_imputed = X.fillna(self.fillna_values)

        # Separar colunas numéricas e categóricas
        num_cols = [f for f in self.initial_features if f not in self.initial_cat_features]
        X_num = X_imputed[num_cols]
        X_cat = X_imputed[self.initial_cat_features]

        # Fit e transform do MeanMapper
        self.mean_mapper.fit(X_cat, y)
        X_cat_encoded = self.mean_mapper.transform(X_cat)

        # Combinar numérico + categórico encodado
        X_combined = pd.concat([X_num, X_cat_encoded], axis=1)
        self.final_features_order = X_combined.columns.tolist()

        # Fit do StandardScaler
        self.scaler.fit(X_combined)
        return self

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica a transformação aprendida no fit (imputação → encoding → scaling).

        Args:
            X_raw (pd.DataFrame): Dados a transformar (treino ou teste).

        Returns:
            pd.DataFrame: Dados transformados, prontos para entrada nos modelos.
        """
        X = X_raw[self.initial_features].copy()
        X_imputed = X.fillna(self.fillna_values)

        num_cols = [f for f in self.initial_features if f not in self.initial_cat_features]
        X_num = X_imputed[num_cols]
        X_cat = X_imputed[self.initial_cat_features]

        X_cat_encoded = self.mean_mapper.transform(X_cat)
        X_combined = pd.concat([X_num, X_cat_encoded], axis=1)

        # Garantir a mesma ordem de colunas do fit
        X_ordered = X_combined[self.final_features_order]

        X_scaled_array = self.scaler.transform(X_ordered)
        return pd.DataFrame(X_scaled_array, columns=self.final_features_order, index=X_raw.index)


# ==============================================================================
# 4. UTILITÁRIOS DE SELEÇÃO DE VARIÁVEIS E SERIALIZAÇÃO
# ==============================================================================

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o Variance Inflation Factor (VIF) para cada coluna de X.

    Args:
        X (pd.DataFrame): DataFrame com features numéricas (deve incluir a constante
                          se o modelo usar intercepto).

    Returns:
        pd.DataFrame: DataFrame com colunas 'feature' e 'VIF'.
    """
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [
        variance_inflation_factor(X.values, i) for i in range(len(X.columns))
    ]
    return vif_data


def iterative_vif_removal(
    X: pd.DataFrame,
    threshold: float = 5.0
) -> tuple[pd.DataFrame, list]:
    """
    Remove iterativamente a variável com o maior VIF enquanto ele exceder o threshold.
    A constante ('const') nunca é removida.

    EXCLUSIVO do pipeline OLS — não deve ser aplicado a XGBoost/KNN,
    pois esses métodos não-paramétricos toleram multicolinearidade.

    Args:
        X (pd.DataFrame): DataFrame com features + constante (após sm.add_constant).
        threshold (float): VIF máximo aceitável. Padrão: 5.0.

    Returns:
        tuple: (X_reduzido, lista_de_features_removidas)
    """
    X_current = X.copy()
    removed = []

    print(f"\n  Iniciando remoção iterativa por VIF (threshold={threshold})...")
    while True:
        vif_df = calculate_vif(X_current)
        max_vif = vif_df['VIF'].max()
        max_feature = vif_df.loc[vif_df['VIF'] == max_vif, 'feature'].iloc[0]

        if max_vif > threshold and max_feature != 'const':
            removed.append(max_feature)
            print(f"  Removendo '{max_feature}' (VIF={max_vif:.2f})")
            X_current = X_current.drop(columns=[max_feature])
        else:
            break

    print(f"  Features removidas por VIF: {removed if removed else 'nenhuma'}")
    return X_current, removed


def iterative_pvalue_removal(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    threshold: float = 0.15
) -> tuple:
    """
    Backward selection: remove iterativamente a variável com o maior p-valor enquanto
    ele exceder o threshold. A constante ('const') nunca é removida.

    EXCLUSIVO do pipeline OLS — não deve ser aplicado a XGBoost/KNN.

    Args:
        X_train (pd.DataFrame): Features de treino (com constante).
        y_train (pd.Series): Target de treino.
        X_test (pd.DataFrame): Features de teste (as mesmas remoções são espelhadas).
        threshold (float): P-valor máximo aceitável. Padrão: 0.15.

    Returns:
        tuple: (ols_model_final, X_train_selecionado, X_test_selecionado, features_removidas)
    """
    current_X_train = X_train.copy()
    current_X_test = X_test.copy()
    removed = []

    print(f"\n  Iniciando backward selection por p-valor (threshold={threshold})...")
    while True:
        model = sm.OLS(y_train, current_X_train).fit()
        p_values = model.pvalues.drop('const', errors='ignore')

        if p_values.empty:
            break

        max_p_feature = p_values.idxmax()
        max_p = p_values.max()

        if max_p > threshold:
            removed.append(max_p_feature)
            print(f"  Removendo '{max_p_feature}' (p={max_p:.4f}, R²={model.rsquared:.4f})")
            current_X_train = current_X_train.drop(columns=[max_p_feature])
            current_X_test = current_X_test.drop(columns=[max_p_feature])
        else:
            break

    print(f"  Features removidas por p-valor: {removed if removed else 'nenhuma'}")
    return model, current_X_train, current_X_test, removed


def save_model(obj, path: str, name: str) -> None:
    """
    Serializa qualquer objeto Python com cloudpickle e salva em disco.

    Args:
        obj: Objeto a serializar (modelo, pipeline, função, etc.).
        path (str): Caminho completo do arquivo .pkl de destino.
        name (str): Nome descritivo exibido no log.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  Salvo: '{name}' -> '{path}'")
