"""
python/run_rentabilidade.py
---------------------------
Pipeline de cálculo de rentabilidade por anúncio.

Responsável por:
    1. Carregar modelos de preço e disponibilidade + pipeline de pré-processamento
    2. Sanity check: validar o modelo de preço na base AirBnb (métricas)
    3. Aplicar modelos na base QuintoAndar:
       3a. Previsão de preço  → coluna 'Aluguel estimado'
       3b. Previsão de dias alugados por imóvel (via modelo de disponibilidade)
    4. Simular receita anual via Monte Carlo (com centros por imóvel ou global)
    5. Otimizar o portfólio de R$ 10M via MIP/PuLP
    6. Salvar os resultados processados para uso no notebook de visualização

=============================================================================
PARÂMETROS CONFIGURÁVEIS
=============================================================================
Ajuste as constantes abaixo para experimentar diferentes cenários:
"""

import sys
import os
import numpy as np
import pandas as pd
import cloudpickle as pickle
import xgboost
import statsmodels.regression.linear_model as sm_lm
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.optimization import simulate_annual_revenue, optimize_portfolio
from utils.modeling import create_target_derived_features
from utils.macro import fetch_macro_rates

# ==============================================================================
# PARÂMETROS DA SIMULAÇÃO DE MONTE CARLO
# ==============================================================================
N_SIMULATIONS    = 1000   # Número de iterações por imóvel
MIN_RENTED_DAYS  = 0.0    # Piso de dias alugados
MAX_RENTED_DAYS  = 365.0  # Teto de dias alugados
RANDOM_SEED      = 42     # Semente para reprodutibilidade

# ==============================================================================
# PARÂMETROS DA SIMULAÇÃO GUIADA PELO MODELO DE DISPONIBILIDADE
# ==============================================================================
# Se USE_MODEL_FOR_DAYS = True, o modelo 'best_xgb_model_disp' fornece a média
# individualizada de dias alugados para cada imóvel (captura heterogeneidade
# geográfica e de características). A simulação em torno dessa média
# representa a incerteza do modelo.
#
# Se USE_MODEL_FOR_DAYS = False, todos os imóveis recebem a mesma média global
# (comportamento antigo).
# ==============================================================================
USE_MODEL_FOR_DAYS = True   # True: centros por imóvel via modelo | False: média global
STD_DIAS_MODELO    = 50.0   # Desvio padrão em torno da predição do modelo.
                             # Deve refletir a incerteza do modelo (referência: RMSE no teste).
                             # Quanto maior, mais larga a distribuição de cenários.

# Parâmetros usados apenas se USE_MODEL_FOR_DAYS = False (média global)
MEAN_RENTED_DAYS = 226.0  # Média global de dias alugados
STD_RENTED_DAYS  = 70.0   # Desvio padrão global

# ==============================================================================
# PARÂMETROS DO MODELO DE OTIMIZAÇÃO (PERFIL DE RISCO)
# Devem somar 1.0. Ajuste para refletir o perfil do investidor:
#   Conservador → aumentar PESO_P5
#   Moderado    → pesos equilibrados (padrão abaixo)
#   Agressivo   → aumentar PESO_P95
# ==============================================================================
ORCAMENTO   = 10_000_000  # Orçamento total disponível (R$)
PESO_P5     = 0.30        # Peso do cenário pessimista  (percentil 5)
PESO_MEDIA  = 0.50        # Peso do cenário médio       (média das simulações)
PESO_P95    = 0.20        # Peso do cenário otimista    (percentil 95)

# ==============================================================================
# PARÂMETROS DO SOLVER MIP
# ==============================================================================
MIP_TIME_LIMIT = 60    # Limite de tempo do solver CBC (segundos)
MIP_GAP_REL    = 0.01  # Tolerância de gap relativo (1% = aceitar solução 99% ótima)

# ==============================================================================
# PARÂMETROS MACROECONÔMICOS
# ==============================================================================
# Controla como as taxas macroeconômicas são obtidas para o desconto do VPL.
#
# USO_API_MACRO = True  → tenta buscar Selic/IPCA via python-bcb (SGS/Focus BCB).
#                         Em caso de falha, usa os valores de fallback abaixo.
# USO_API_MACRO = False → usa apenas os valores de fallback (útil offline/CI).
#
# Os valores de fallback estão em utils/macro.py (FALLBACK_SELIC_ANUAL, etc.).
# ==============================================================================
USO_API_MACRO      = True    # Habilitar consulta à API do Banco Central
MACRO_SPREAD       = 0.025   # Spread imobiliário sobre a Selic (decimal, ex: 0.025 = 2,5 p.p.)
MACRO_HORIZONTE    = 12      # Horizonte de análise em meses (alinhado ao simulate_annual_revenue)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("  Pipeline de Rentabilidade por Anuncio")
    print("=" * 70)

    # Caminhos
    root           = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path_processed = os.path.join(root, 'data', 'processed')
    path_models    = os.path.join(root, 'models')

    # ==========================================================================
    # [1/5] CARREGAR MODELOS E DADOS
    # ==========================================================================
    print("\n[1/5] Carregando modelos e bases de dados...")

    def load_pkl(filename):
        with open(os.path.join(path_models, filename), 'rb') as f:
            return pickle.load(f)

    pipeline    = load_pkl('full_preprocessing_pipeline.pkl')
    best_model  = load_pkl('best_xgb_model.pkl')
    model_disp  = load_pkl('best_xgb_model_disp.pkl')  # Modelo de disponibilidade

    df_airbnb      = pd.read_csv(os.path.join(path_processed, 'base_features_AirBnb.csv'), index_col=0)
    df_quintoAndar = pd.read_csv(os.path.join(path_processed, 'base_features_quintoAndar.csv'), index_col=0)

    print(f"  AirBnb:      {len(df_airbnb):,} registros")
    print(f"  QuintoAndar: {len(df_quintoAndar):,} registros")
    print(f"  Modelo preco: {type(best_model).__name__}")
    print(f"  Modelo disp:  {type(model_disp).__name__} ({len(model_disp.feature_names_in_)} features)")
    print(f"  Modo simulacao: {'Guiado pelo modelo' if USE_MODEL_FOR_DAYS else 'Media global'}")

    # Extrair nomes das features do modelo carregado
    if isinstance(best_model, xgboost.XGBRegressor):
        model_features = list(best_model.feature_names_in_)
    elif isinstance(best_model, sm_lm.RegressionResultsWrapper):
        model_features = list(best_model.model.exog_names)
    else:
        raise TypeError(f"Tipo de modelo nao suportado: {type(best_model)}")

    print(f"  Features do modelo ({len(model_features)}): {model_features}")

    # ==========================================================================
    # [2/5] SANITY CHECK — VALIDAÇÃO DO MODELO NA BASE AIRBNB
    # ==========================================================================
    print("\n[2/5] Sanity check: validando modelo na base AirBnb...")

    # Pré-processar a base AirBnb com o pipeline treinado
    df_airbnb_proc = pipeline.transform(df_airbnb)
    if isinstance(best_model, xgboost.XGBRegressor):
        df_airbnb_proc = df_airbnb_proc[model_features]
    elif isinstance(best_model, sm_lm.RegressionResultsWrapper):
        df_airbnb_proc = sm.add_constant(df_airbnb_proc)[model_features]

    y_true = df_airbnb['Preço do aluguel (escala log)']
    y_pred = best_model.predict(df_airbnb_proc)

    mse  = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(np.expm1(y_true), np.expm1(y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE (escala original): R$ {mae:.2f}")
    print(f"  R2:   {r2:.4f}")

    # Salvar predições AirBnb para o notebook de visualização
    df_sanity = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df_sanity.to_csv(os.path.join(path_processed, 'sanity_check_airbnb.csv'), index=True)
    print(f"  Predicoes salvas: sanity_check_airbnb.csv")

    # ==========================================================================
    # [3/5] PREDIÇÃO — APLICAR MODELO NA BASE QUINTOANDAR
    # ==========================================================================
    print("\n[3/5] Aplicando modelo na base QuintoAndar...")

    # Pré-processar QuintoAndar (resultado escalado reutilizado na etapa de disponibilidade)
    df_quintoAndar_scaled = pipeline.transform(df_quintoAndar)

    df_quintoAndar_proc = df_quintoAndar_scaled.copy()
    if isinstance(best_model, xgboost.XGBRegressor):
        df_quintoAndar_proc = df_quintoAndar_proc[model_features]
    elif isinstance(best_model, sm_lm.RegressionResultsWrapper):
        df_quintoAndar_proc = sm.add_constant(df_quintoAndar_proc)[model_features]

    # 3a. Predição de preço em escala original (revertendo log1p)
    df_quintoAndar['Aluguel estimado'] = np.expm1(best_model.predict(df_quintoAndar_proc))

    print(f"  Aluguel estimado — média: R$ {df_quintoAndar['Aluguel estimado'].mean():.2f} "
          f"| mediana: R$ {df_quintoAndar['Aluguel estimado'].median():.2f}")

    # 3b. Predição de dias alugados por imóvel (modelo de disponibilidade)
    # O modelo de disponibilidade foi treinado com:
    #   X_train_scaled (todas as features) + Diferenca_preco_bairro + Diferenca_preco_quadrante
    # Para a base QuintoAndar, usamos o log1p do aluguel estimado como proxy do target de preço.
    print("  Prevendo dias alugados por imovel (best_xgb_model_disp)...")

    y_log_qa = np.log1p(df_quintoAndar['Aluguel estimado'])
    y_log_qa.index = df_quintoAndar_scaled.index  # alinhar índice

    derived_feats = create_target_derived_features(
        X_data=df_quintoAndar_scaled,
        y_data=y_log_qa,
    )
    X_qa_disp = pd.concat([df_quintoAndar_scaled, derived_feats], axis=1)
    # Garantir ordem exata de features do modelo
    X_qa_disp = X_qa_disp[model_disp.feature_names_in_]

    dias_estimados = model_disp.predict(X_qa_disp)
    dias_estimados = np.clip(dias_estimados, MIN_RENTED_DAYS, MAX_RENTED_DAYS)

    df_quintoAndar['Dias Alugados Estimado'] = dias_estimados

    print(f"  Dias alugados estimado — média: {dias_estimados.mean():.1f} "
          f"| mediana: {np.median(dias_estimados):.1f} "
          f"| min: {dias_estimados.min():.1f} | max: {dias_estimados.max():.1f}")

    # ==========================================================================
    # [4/5] SIMULAÇÃO DE MONTE CARLO — RECEITA ANUAL
    # ==========================================================================
    print("\n[4/5] Simulando receita anual (Monte Carlo)...")

    if USE_MODEL_FOR_DAYS:
        # Abordagem guiada pelo modelo: centro individualizado por imóvel
        # O modelo captura heterogeneidade geográfica e de características.
        # O std representa a incerteza do modelo em torno de cada predição.
        mean_days = df_quintoAndar['Dias Alugados Estimado'].values
        std_days  = STD_DIAS_MODELO
        print(f"  Modo: GUIADO PELO MODELO (dias estimados por imovel)")
        print(f"  Std (incerteza do modelo): {std_days} dias | N: {N_SIMULATIONS}")
        print(f"  Centro por imovel — media: {mean_days.mean():.1f} | "
              f"min: {mean_days.min():.1f} | max: {mean_days.max():.1f}")
    else:
        # Abordagem global: todos os imóveis recebem a mesma distribuição
        mean_days = MEAN_RENTED_DAYS
        std_days  = STD_RENTED_DAYS
        print(f"  Modo: MEDIA GLOBAL ({mean_days} dias, std={std_days})")

    df_quintoAndar = simulate_annual_revenue(
        df=df_quintoAndar,
        daily_rent_col='Aluguel estimado',
        n_simulations=N_SIMULATIONS,
        mean_rented_days=mean_days,
        std_rented_days=std_days,
        min_rented_days=MIN_RENTED_DAYS,
        max_rented_days=MAX_RENTED_DAYS,
        random_seed=RANDOM_SEED,
    )

    # Salvar base QuintoAndar enriquecida (com aluguel + simulação)
    df_quintoAndar.to_csv(os.path.join(path_processed, 'base_quintoAndar_rentabilidade.csv'), index=True)
    print(f"  Base enriquecida salva: base_quintoAndar_rentabilidade.csv")

    # ==========================================================================
    # [5/5] OTIMIZAÇÃO MIP — PORTFÓLIO ÓTIMO
    # ==========================================================================
    print("\n[5/5] Otimizando portfolio de investimentos (MIP)...")
    print(f"  Orcamento: R$ {ORCAMENTO:,.2f}")
    print(f"  Perfil de risco — p5: {PESO_P5} | media: {PESO_MEDIA} | p95: {PESO_P95}")

    # --- Taxas macroeconômicas -------------------------------------------
    print("\n  [macro] Buscando taxas macroeconomicas...")
    if USO_API_MACRO:
        macro = fetch_macro_rates(
            horizonte_meses=MACRO_HORIZONTE,
            spread_imovel=MACRO_SPREAD,
        )
    else:
        from utils.macro import MacroRates, FALLBACK_SELIC_ANUAL, FALLBACK_IPCA_ANUAL
        macro = MacroRates(
            selic_anual=FALLBACK_SELIC_ANUAL,
            ipca_anual=FALLBACK_IPCA_ANUAL,
            spread_imovel=MACRO_SPREAD,
            horizonte_meses=MACRO_HORIZONTE,
            fonte="fallback_forcado",
        )

    print(f"  {macro}")
    print(f"  Fonte: {macro.fonte}")
    print(f"  Taxa nominal (Selic+spread): {macro.taxa_desconto_nominal:.2%} a.a.")
    print(f"  Taxa real (Fisher):          {macro.taxa_desconto_real:.2%} a.a.")
    print(f"  Fator VP (anuidade {macro.horizonte_meses}m):      {macro.fator_vp:.4f}")
    # ----------------------------------------------------------------------

    resultado = optimize_portfolio(
        df=df_quintoAndar,
        budget=ORCAMENTO,
        sale_col='Valor de venda',
        peso_p5=PESO_P5,
        peso_media=PESO_MEDIA,
        peso_p95=PESO_P95,
        time_limit_seconds=MIP_TIME_LIMIT,
        gap_rel=MIP_GAP_REL,
        macro_rates=macro,
    )

    # Salvar portfólio ótimo
    if not resultado['df_portfolio'].empty:
        resultado['df_portfolio'].to_csv(
            os.path.join(path_processed, 'portfolio_otimo.csv'), index=True
        )
        print(f"  Portfolio salvo: portfolio_otimo.csv")

    # Salvar todos os candidatos (selecionados + rejeitados) para comparação no notebook
    candidatos_cols = [
        c for c in [
            'url', 'Bairro', 'Valor de venda',
            'Receita Anual p5', 'Receita Anual Media', 'Receita Anual p95',
            'Retorno Esperado Ajustado', 'VPL Estimado', 'Selecionado',
        ]
        if c in resultado['df_opt'].columns
    ]
    resultado['df_opt'][candidatos_cols].to_csv(
        os.path.join(path_processed, 'portfolio_candidatos.csv'), index=True
    )
    print(f"  Candidatos completos salvos: portfolio_candidatos.csv ({len(resultado['df_opt']):,} imoveis)")

    # Salvar metadados da otimização (para o notebook de visualização)
    meta = {
        'status':                  resultado['status'],
        'n_imoveis_selecionados':  len(resultado['imoveis_selecionados']),
        'custo_total':             resultado['custo_total'],
        'retorno_total_esperado':  resultado['retorno_total_esperado'],
        'orcamento':               ORCAMENTO,
        'peso_p5':                 PESO_P5,
        'peso_media':              PESO_MEDIA,
        'peso_p95':                PESO_P95,
        # --- Taxas macroeconômicas ---
        'macro_fonte':             macro.fonte,
        'selic_anual':             macro.selic_anual,
        'ipca_anual':              macro.ipca_anual,
        'spread_imovel':           macro.spread_imovel,
        'taxa_desconto_nominal':   macro.taxa_desconto_nominal,
        'taxa_desconto_real':      macro.taxa_desconto_real,
        'fator_vp':                macro.fator_vp,
        'horizonte_meses':         macro.horizonte_meses,
    }
    pd.Series(meta).to_csv(os.path.join(path_processed, 'portfolio_meta.csv'), header=False)
    print(f"  Metadados salvos: portfolio_meta.csv")

    # ==========================================================================
    # SUMÁRIO FINAL
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  Pipeline concluido com sucesso!")
    print("=" * 70)
    print(f"  Modelo validado   — R2: {r2:.4f} | MAE: R$ {mae:.2f}")
    print(f"  Imoveis avaliados — {len(df_quintoAndar):,} (QuintoAndar)")
    print(f"  Portfolio otimo   — {len(resultado['imoveis_selecionados'])} imoveis | "
          f"Investimento: R$ {resultado['custo_total']:,.2f} | "
          f"Retorno esperado: R$ {resultado['retorno_total_esperado']:,.2f}")
    print("=" * 70)
    print(f"\n  Arquivos salvos em: {path_processed}")


if __name__ == '__main__':
    main()
