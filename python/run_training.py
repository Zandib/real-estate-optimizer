"""
python/run_training.py
----------------------
Script de treinamento de modelos preditivos para o projeto AirBnb.

Responsável por:
    - Carregar a base já processada (data/processed/base_features_AirBnb.csv)
    - Executar o pipeline de pré-processamento
    - Treinar cada modelo com seu próprio conjunto de features e etapas
    - Salvar os modelos treinados em models/

=============================================================================
ARQUITETURA DE PIPELINES (cada modelo tem etapas específicas):
=============================================================================

  MODELO OLS (Regressão Linear)
  ─────────────────────────────
  Base:            Entire home/apt, disponibilidade > 0 e < 365
  Target:          Preço do aluguel (escala log)
  Features:        Latitude, Longitude, + numéricas + mean-encoded categóricas
  Etapas extras:   ✅ Remoção iterativa por VIF (threshold = 5)
                   ✅ Backward selection por p-valor (threshold = 0.15)
  Scaler:          StandardScaler (aplicado após seleção de variáveis)

  MODELO XGBoost — Preço
  ──────────────────────
  Base:            Mesmo filtro do OLS
  Target:          Preço do aluguel (escala log)
  Features:        Todas as features (sem remoção por VIF/p-valor)
  Etapas extras:   ✅ GridSearchCV (3-fold, otimizando MSE)
  Scaler:          StandardScaler (mesmas features do OLS base, sem seleção)

  MODELO KNN — Preço
  ──────────────────
  Base:            Mesmo filtro do OLS
  Target:          Preço do aluguel (escala log)
  Features:        Todas as features (sem remoção por VIF/p-valor)
  Etapas extras:   ✅ GridSearchCV (3-fold, otimizando MSE)
  Scaler:          StandardScaler (mesmo espaço do XGBoost)

  MODELO XGBoost — Disponibilidade
  ─────────────────────────────────
  Base:            Mesmo filtro do OLS
  Target:          Dias alugados (estimado) — variável de ocupação
  Features:        Todas as features de preço + features derivadas de target
                   ('Diferenca_preco_bairro', 'Diferenca_preco_quadrante')
  Etapas extras:   ✅ create_target_derived_features (feature engineering)
                   ✅ GridSearchCV (3-fold, otimizando MSE)
  Scaler:          StandardScaler (mesmo da base de preço, + novas features)

=============================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Adicionar a pasta raiz ao sys.path para importar o pacote utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.modeling import (
    CustomPreprocessingPipeline,
    create_target_derived_features,
    calculate_vif,
    iterative_vif_removal,
    iterative_pvalue_removal,
    save_model,
)



# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("  Pipeline de Treinamento de Modelos — AirBnb")
    print("=" * 70)

    # --- Caminhos ---
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path_processed = os.path.join(root, 'data', 'processed')
    path_models = os.path.join(root, 'models')
    os.makedirs(path_models, exist_ok=True)

    # ==========================================================================
    # [1/6] CARGA E FILTRAGEM DOS DADOS
    # ==========================================================================
    print("\n[1/6] Carregando e filtrando a base AirBnb processada...")

    df_base = pd.read_csv(os.path.join(path_processed, 'base_features_AirBnb.csv'), index_col=0)

    # Filtros padrão aplicados por todos os modelos
    df = df_base.copy()
    df = df.loc[df['Tipo de locação'] == 'Entire home/apt'].reset_index(drop=True)
    df = df.loc[df['Disponibilidade no ano'] < 365].reset_index(drop=True)
    df = df.loc[df['Disponibilidade no ano'] > 0].reset_index(drop=True)

    print(f"  Base filtrada: {len(df)} observações")

    # ==========================================================================
    # [2/6] DEFINIÇÃO DE FEATURES E TARGETS
    # ==========================================================================
    print("\n[2/6] Definindo features e variáveis alvo...")

    features = [
        'Latitude', 'Longitude', 'Banheiros', 'Quartos',
        'Distancia da praia', 'Distância minima de atracao turistica',
        'Distância minima de entrada de metrô', 'Distância minima de escola',
        'Bairro', 'quadrante'
    ]
    target_preco = 'Preço do aluguel (escala log)'
    target_disp = 'Dias alugados (estimado)'

    # Identificar features categóricas e numéricas
    cat_features = []
    num_features = []
    for f in features:
        if not pd.api.types.is_numeric_dtype(df[f]):
            # Catches object, category, pd.StringDtype (pandas 2.x), and any
            # other non-numeric dtype that may arise depending on how the CSV was read.
            df[f] = df[f].astype('category')
            cat_features.append(f)
        else:
            df[f] = df[f].astype('float')
            num_features.append(f)

    print(f"  Categóricas: {cat_features}")
    print(f"  Numéricas:   {num_features}")

    # ==========================================================================
    # [3/6] SPLIT TREINO/TESTE E PRÉ-PROCESSAMENTO BASE
    # ==========================================================================
    print("\n[3/6] Realizando train/test split e pré-processamento...")

    X = df[features]
    y_preco = df[target_preco]
    y_disp = df[target_disp]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_preco, test_size=0.2, random_state=42
    )

    # Targets de disponibilidade alinhados com o split de preço
    y_train_disp = y_disp.loc[X_train_raw.index]
    y_test_disp = y_disp.loc[X_test_raw.index]

    # Calcular fillna com médias do treino (para evitar data leakage)
    fillna_dict = {
        'Banheiros': X_train_raw['Banheiros'].mean(),
        'Quartos': X_train_raw['Quartos'].mean(),
        'Distancia da praia': X_train_raw['Distancia da praia'].mean()
    }

    # --- Pipeline base: MeanMapper + StandardScaler (fit apenas no treino) ---
    # Instanciado aqui, na etapa onde o trabalho é feito, para evitar re-fit posterior.
    # XGBoost e KNN consumirão as saídas transformadas deste pipeline.
    # O objeto 'full_pipeline' será salvo no passo [6/6] sem nenhuma recomputação.
    full_pipeline = CustomPreprocessingPipeline(
        initial_features=features,
        initial_cat_features=cat_features,
        fillna_values=fillna_dict
    )
    full_pipeline.fit(X_train_raw, y_train)

    X_train_scaled = full_pipeline.transform(X_train_raw)
    X_test_scaled  = full_pipeline.transform(X_test_raw)

    print(f"  X_train shape: {X_train_scaled.shape} | X_test shape: {X_test_scaled.shape}")

    # ==========================================================================
    # [4/6] MODELO OLS — Pipeline Exclusivo: VIF + Backward p-value + OLS
    # ==========================================================================
    print("\n[4/6] Treinando modelo OLS (com VIF e seleção por p-valor)...")
    print("  ── Esta etapa é EXCLUSIVA do OLS. XGBoost e KNN não passam por ela. ──")

    # Adicionar constante para OLS (intercepto)
    X_train_ols = sm.add_constant(X_train_scaled.copy())
    X_test_ols = sm.add_constant(X_test_scaled.copy())

    # Remover Latitude e Longitude antes do VIF (coordenadas são mantidas como features
    # mas causam alta multicolinearidade no VIF — são usadas no modelo final se p-valor ok)
    X_train_vif = X_train_ols.drop(columns=['Latitude', 'Longitude'], errors='ignore')
    X_train_vif_reduced, vif_removed = iterative_vif_removal(X_train_vif, threshold=5.0)

    # Restaurar Latitude e Longitude (removidas temporariamente para o cálculo do VIF)
    cols_after_vif = [c for c in X_train_ols.columns if c not in vif_removed or c in ['Latitude', 'Longitude']]
    X_train_ols_reduced = X_train_ols[cols_after_vif]
    X_test_ols_reduced = X_test_ols[cols_after_vif]

    # Backward selection por p-valor
    ols_model, X_train_ols_final, X_test_ols_final, pval_removed = iterative_pvalue_removal(
        X_train_ols_reduced, y_train, X_test_ols_reduced, threshold=0.15
    )

    # Métricas do OLS
    y_pred_ols = ols_model.predict(X_test_ols_final)
    mae_ols = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_ols))
    r2_ols = r2_score(y_test, y_pred_ols)
    print(f"\n  OLS — R²: {r2_ols:.4f} | MAE (escala original): R${mae_ols:.2f}")
    print(f"  Features finais no OLS: {list(X_train_ols_final.columns)}")

    # Salvar modelo OLS
    save_model(ols_model, os.path.join(path_models, 'ols_model.pkl'), 'ols_model')

    # ==========================================================================
    # [5/6] MODELOS XGBoost e KNN — Pipeline SEM seleção de variáveis
    # ==========================================================================
    print("\n[5/6] Treinando XGBoost e KNN (sem VIF/p-valor — todos as features)...")
    print("  ── XGBoost e KNN usam X_train_scaled completo (sem remoção de variáveis). ──")

    # ------------------------------------------------------------------
    # 5a. XGBoost — Preço
    # ------------------------------------------------------------------
    print("\n  [5a] XGBoost — Preço do aluguel (GridSearchCV)...")
    param_grid_xgb = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9]
    }
    grid_xgb = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid_xgb,
        scoring='neg_mean_squared_error',
        cv=3, n_jobs=-1, verbose=1
    )
    grid_xgb.fit(X_train_scaled, y_train)
    best_xgb_model = grid_xgb.best_estimator_

    y_pred_xgb = best_xgb_model.predict(X_test_scaled)
    mae_xgb = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    print(f"  XGBoost Preço — R²: {r2_xgb:.4f} | MAE: R${mae_xgb:.2f} | Params: {grid_xgb.best_params_}")
    save_model(best_xgb_model, os.path.join(path_models, 'best_xgb_model.pkl'), 'best_xgb_model')

    # ------------------------------------------------------------------
    # 5b. KNN — Preço
    # ------------------------------------------------------------------
    print("\n  [5b] KNN — Preço do aluguel (GridSearchCV)...")
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    grid_knn = GridSearchCV(
        estimator=KNeighborsRegressor(),
        param_grid=param_grid_knn,
        scoring='neg_mean_squared_error',
        cv=3, n_jobs=-1, verbose=1
    )
    grid_knn.fit(X_train_scaled, y_train)
    best_knn_model = grid_knn.best_estimator_

    y_pred_knn = best_knn_model.predict(X_test_scaled)
    mae_knn = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_knn))
    r2_knn = r2_score(y_test, y_pred_knn)
    print(f"  KNN Preço — R²: {r2_knn:.4f} | MAE: R${mae_knn:.2f} | Params: {grid_knn.best_params_}")
    save_model(best_knn_model, os.path.join(path_models, 'best_knn_model.pkl'), 'best_knn_model')

    # ------------------------------------------------------------------
    # 5c. XGBoost — Disponibilidade (com features derivadas de target)
    # ------------------------------------------------------------------
    print("\n  [5c] XGBoost — Disponibilidade (com create_target_derived_features)...")
    print("  ── Feature engineering de disponibilidade: adicionando diferenças de preço. ──")

    new_feats_train = create_target_derived_features(X_train_scaled, y_train)
    new_feats_test = create_target_derived_features(X_test_scaled, y_test)

    X_train_disp = pd.concat([X_train_scaled, new_feats_train], axis=1)
    X_test_disp = pd.concat([X_test_scaled, new_feats_test], axis=1)

    param_grid_disp = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9]
    }
    grid_xgb_disp = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid_disp,
        scoring='neg_mean_squared_error',
        cv=3, n_jobs=-1, verbose=1
    )
    grid_xgb_disp.fit(X_train_disp, y_train_disp)
    best_xgb_model_disp = grid_xgb_disp.best_estimator_

    y_pred_xgb_disp = best_xgb_model_disp.predict(X_test_disp)
    mae_xgb_disp = mean_absolute_error(y_test_disp, y_pred_xgb_disp)
    r2_xgb_disp = r2_score(y_test_disp, y_pred_xgb_disp)
    print(f"  XGBoost Disp — R²: {r2_xgb_disp:.4f} | MAE: {mae_xgb_disp:.2f} dias | Params: {grid_xgb_disp.best_params_}")
    save_model(best_xgb_model_disp, os.path.join(path_models, 'best_xgb_model_disp.pkl'), 'best_xgb_model_disp')

    # Salvar a função de feature engineering (para reprodução no notebook de resultados)
    save_model(create_target_derived_features,
               os.path.join(path_models, 'create_target_derived_features.pkl'),
               'create_target_derived_features')

    # ==========================================================================
    # [6/6] SALVAR PIPELINE DE PRÉ-PROCESSAMENTO + DADOS DE TREINO/TESTE
    # ==========================================================================
    print("\n[6/6] Salvando pipeline de pré-processamento e splits de dados...")

    # Pipeline já treinado no passo [3/6] — apenas serializar, sem re-fit.
    save_model(full_pipeline, os.path.join(path_models, 'full_preprocessing_pipeline.pkl'),
               'full_preprocessing_pipeline')

    # Salvar splits processados (X escalado + y) para uso no notebook de análise
    splits = {
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'X_train_disp': X_train_disp, 'X_test_disp': X_test_disp,
        'X_train_ols': X_train_ols_final, 'X_test_ols': X_test_ols_final,
        'y_train': y_train, 'y_test': y_test,
        'y_train_disp': y_train_disp, 'y_test_disp': y_test_disp
    }
    for name, obj in splits.items():
        path_csv = os.path.join(root, 'data', 'processed', f'{name}.csv')
        obj.to_csv(path_csv, index=True)
        print(f"  💾 '{name}.csv' salvo")

    # ==========================================================================
    # SUMÁRIO FINAL
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  ✅ Treinamento concluído! Resumo de desempenho:")
    print("=" * 70)
    print(f"  {'Modelo':<35} {'R²':>8} {'MAE':>12}")
    print(f"  {'-'*55}")
    print(f"  {'OLS (Preço - log)':<35} {r2_ols:>8.4f} {'R$' + f'{mae_ols:.2f}':>12}")
    print(f"  {'XGBoost (Preço - log)':<35} {r2_xgb:>8.4f} {'R$' + f'{mae_xgb:.2f}':>12}")
    print(f"  {'KNN (Preço - log)':<35} {r2_knn:>8.4f} {'R$' + f'{mae_knn:.2f}':>12}")
    print(f"  {'XGBoost (Disponibilidade)':<35} {r2_xgb_disp:>8.4f} {f'{mae_xgb_disp:.2f} dias':>12}")
    print("=" * 70)
    print(f"\n  Modelos salvos em: {path_models}")


if __name__ == '__main__':
    main()
