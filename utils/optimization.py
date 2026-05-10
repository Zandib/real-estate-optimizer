"""
utils/optimization.py
---------------------
Módulo com funções para simulação de receita e otimização de portfólio imobiliário.

Conteúdo:
    - simulate_annual_revenue: Simulação de Monte Carlo para receita anual bruta.
    - optimize_portfolio: Otimização de portfólio via Programação Inteira Mista (MIP/PuLP),
                          com suporte a desconto macroeconômico via VPL (Valor Presente Líquido).

Integração Macroeconômica:
    Quando um objeto `MacroRates` (de utils.macro) é fornecido a `optimize_portfolio`,
    a função objetivo PuLP passa a maximizar o VPL do fluxo de aluguel descontado
    pela taxa real livre de risco ajustada pelo spread imobiliário:

        i_real = [(1 + Selic + Spread) / (1 + IPCA)] - 1    ← Fisher
        FATOR_VP = [1 − (1 + i_mes)^{−T}] / i_mes            ← anuidade mensal
        VPL[i] = Retorno_Ajustado[i] × FATOR_VP

    O VPL é pré-calculado por imóvel como escalar, preservando a linearidade do MIP.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pulp

from utils.macro import MacroRates


# ==============================================================================
# 1. SIMULATE_ANNUAL_REVENUE — Simulação de Monte Carlo
# ==============================================================================

def simulate_annual_revenue(
    df: pd.DataFrame,
    daily_rent_col: str = 'Aluguel estimado',
    n_simulations: int = 1000,
    mean_rented_days = 226.0,
    std_rented_days: float = 70.0,
    min_rented_days: float = 0.0,
    max_rented_days: float = 365.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Simula a receita anual bruta de cada imóvel via Monte Carlo.

    Para cada imóvel, gera `n_simulations` amostras de dias alugados no ano
    a partir de uma distribuição Normal truncada. A receita de cada simulação
    é calculada como: aluguel_diário × dias_alugados. As estatísticas resultantes
    são adicionadas ao DataFrame como novas colunas.

    Args:
        df (pd.DataFrame): Base de imóveis com a coluna de aluguel diário estimado.
        daily_rent_col (str): Nome da coluna com o aluguel diário estimado.
        n_simulations (int): Número de iterações de Monte Carlo por imóvel.
        mean_rented_days (float | np.ndarray): Média de dias alugados por ano.
            - float: mesma média global para todos os imóveis (abordagem simples).
            - np.ndarray de shape (n,): média individualizada por imóvel
              (abordagem guiada pelo modelo — use as predições de best_xgb_model_disp).
        std_rented_days (float): Desvio padrão dos dias alugados.
            Quando usando o modelo, este parâmetro representa a incerteza do modelo
            (uma boa referência é o RMSE do modelo no conjunto de teste).
        min_rented_days (float): Mínimo de dias alugados (clipping inferior).
        max_rented_days (float): Máximo de dias alugados (clipping superior).
        random_seed (int): Semente para reprodutibilidade.

    Returns:
        pd.DataFrame: Cópia do DataFrame com as colunas de receita anual adicionadas:
                      Receita Anual Media, Std, Mediana, Min, Max, p5, p25, p75, p95.
    """
    np.random.seed(random_seed)
    df_out = df.copy()

    # Array 2D: (n_imoveis, n_simulations)
    daily_rent_arr = df_out[daily_rent_col].values.reshape(-1, 1)

    # mean_rented_days pode ser um escalar global ou um array por propriedade.
    # Quando é array, reshape para (n, 1) para broadcasting correto contra (n, n_simulations).
    if np.isscalar(mean_rented_days):
        loc = mean_rented_days
    else:
        loc = np.array(mean_rented_days).reshape(-1, 1)

    simulated_days = np.random.normal(
        loc=loc,
        scale=std_rented_days,
        size=(len(df_out), n_simulations)
    )
    simulated_days = np.clip(simulated_days, min_rented_days, max_rented_days)

    # Receita = aluguel diário × dias alugados
    revenue_matrix = daily_rent_arr * simulated_days

    df_out['Receita Anual Media']   = np.nanmean(revenue_matrix, axis=1)
    df_out['Receita Anual Std']     = np.nanstd(revenue_matrix, axis=1)
    df_out['Receita Anual Mediana'] = np.nanmedian(revenue_matrix, axis=1)
    df_out['Receita Anual Min']     = np.nanmin(revenue_matrix, axis=1)
    df_out['Receita Anual Max']     = np.nanmax(revenue_matrix, axis=1)
    df_out['Receita Anual p5']      = np.nanpercentile(revenue_matrix, 5, axis=1)
    df_out['Receita Anual p25']     = np.nanpercentile(revenue_matrix, 25, axis=1)
    df_out['Receita Anual p75']     = np.nanpercentile(revenue_matrix, 75, axis=1)
    df_out['Receita Anual p95']     = np.nanpercentile(revenue_matrix, 95, axis=1)

    # Preservar a matriz para análise de distribuições individuais no notebook
    df_out.attrs['revenue_matrix'] = revenue_matrix

    print(f"  Simulacao Monte Carlo concluida: {len(df_out)} imoveis x {n_simulations} simulacoes")
    return df_out


# ==============================================================================
# 2. OPTIMIZE_PORTFOLIO — Programação Inteira Mista (MIP) com PuLP
# ==============================================================================

def optimize_portfolio(
    df: pd.DataFrame,
    budget: float,
    sale_col: str = 'Valor de venda',
    peso_p5: float = 0.30,
    peso_media: float = 0.50,
    peso_p95: float = 0.20,
    time_limit_seconds: int = 60,
    gap_rel: float = 0.01,
    macro_rates: Optional[MacroRates] = None,
) -> dict:
    """
    Resolve o problema de otimização de portfólio imobiliário via MIP.

    Formulação base (sem macro_rates):
        Maximizar:   sum(Retorno_Esperado_Ajustado[i] * x[i])
        Sujeito a:   sum(Custo[i] * x[i]) <= orcamento
                     x[i] in {0, 1}  (variável binária: compra ou não compra)

    Formulação com VPL (quando macro_rates é fornecido):
        A função objetivo usa o VPL do fluxo de aluguel descontado pela taxa real:

            i_nominal = Selic + Spread_Imobiliário
            i_real    = [(1 + i_nominal) / (1 + IPCA)] - 1    ← equação de Fisher
            i_mes     = (1 + i_real)^{1/12} - 1
            FATOR_VP  = [1 - (1 + i_mes)^{-T}] / i_mes        ← fator de anuidade mensal

            VPL[i]    = Retorno_Esperado_Ajustado[i] × FATOR_VP

        O VPL é pré-calculado por imóvel como escalar → preserva linearidade do MIP.

    O Retorno Esperado Ajustado é uma média ponderada dos cenários Monte Carlo:
        Retorno = peso_p5 * p5 + peso_media * Media + peso_p95 * p95

    Os pesos representam o perfil de risco do investidor:
        - Conservador:  peso_p5 alto (mais peso no cenário pessimista)
        - Moderado:     pesos equilibrados (padrão)
        - Agressivo:    peso_p95 alto (mais peso no cenário otimista)

    Args:
        df (pd.DataFrame): Base com colunas de custo e receita simulada.
        budget (float): Orçamento total disponível para investimento.
        sale_col (str): Coluna com o valor de venda (custo) de cada imóvel.
        peso_p5 (float): Peso do cenário pessimista (p5) no retorno esperado.
        peso_media (float): Peso do cenário médio no retorno esperado.
        peso_p95 (float): Peso do cenário otimista (p95) no retorno esperado.
        time_limit_seconds (int): Limite de tempo para o solver CBC.
        gap_rel (float): Tolerância de gap relativo para o solver.
        macro_rates (MacroRates | None): Taxas macroeconômicas (Selic, IPCA, spread).
            Se None, mantém comportamento original (maximiza retorno bruto anual).
            Se fornecido, a função objetivo maximiza o VPL descontado pela taxa real.

    Returns:
        dict com:
            - 'status': status da otimização ('Optimal', 'Feasible', etc.)
            - 'imoveis_selecionados': lista de índices dos imóveis comprados
            - 'custo_total': soma dos valores de venda dos imóveis selecionados
            - 'retorno_total_esperado': soma dos coeficientes objetivo dos selecionados
                                        (retorno bruto se macro_rates=None, VPL caso contrário)
            - 'df_portfolio': DataFrame filtrado com os imóveis selecionados
            - 'df_opt': DataFrame completo usado na otimização
            - 'macro_rates': MacroRates utilizado (ou None)
    """
    assert abs(peso_p5 + peso_media + peso_p95 - 1.0) < 1e-6, \
        "Os pesos de risco devem somar 1.0"

    # Filtrar imóveis com dados válidos
    required_cols = [sale_col, 'Receita Anual p5', 'Receita Anual Media', 'Receita Anual p95']
    df_opt = df.dropna(subset=required_cols).copy()
    df_opt = df_opt[df_opt[sale_col] > 0]

    # Calcular retorno esperado ajustado ao perfil de risco (média ponderada dos cenários)
    df_opt['Retorno Esperado Ajustado'] = (
        peso_p5   * df_opt['Receita Anual p5'] +
        peso_media * df_opt['Receita Anual Media'] +
        peso_p95  * df_opt['Receita Anual p95']
    )

    # ------------------------------------------------------------------
    # Desconto macroeconômico — VPL (Valor Presente Líquido)
    # ------------------------------------------------------------------
    # Quando macro_rates é fornecido, o coeficiente da função objetivo PuLP
    # passa a ser o VPL do fluxo de aluguel descontado pela taxa real:
    #   VPL[i] = Retorno_Ajustado[i] × FATOR_VP
    #
    # O FATOR_VP é < 1 quando a taxa real > 0, refletindo o custo de oportunidade
    # do capital e a erosão pelo IPCA. Como é um escalar pré-calculado por imóvel,
    # a linearidade do MIP é preservada.
    # ------------------------------------------------------------------
    if macro_rates is not None:
        df_opt['VPL Estimado'] = df_opt['Retorno Esperado Ajustado'] * macro_rates.fator_vp
        coeficiente_objetivo   = 'VPL Estimado'
        modo_objetivo          = f"VPL (i_real={macro_rates.taxa_desconto_real:.2%}, FatorVP={macro_rates.fator_vp:.4f})"
    else:
        coeficiente_objetivo   = 'Retorno Esperado Ajustado'
        modo_objetivo          = "Retorno Bruto Anual (sem desconto)"

    imoveis_ids = df_opt.index.tolist()
    custos   = df_opt[sale_col].to_dict()
    retornos = df_opt[coeficiente_objetivo].to_dict()

    print(f"  Iniciando otimizacao MIP para {len(imoveis_ids)} imoveis candidatos...")
    print(f"  Modo objetivo: {modo_objetivo}")

    # Definir e resolver o problema MIP
    prob = pulp.LpProblem("Otimizacao_Portfolio_Imoveis", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("comprar", imoveis_ids, cat='Binary')

    prob += pulp.lpSum([retornos[i] * x[i] for i in imoveis_ids]), "Max_Retorno_Objetivo"
    prob += pulp.lpSum([custos[i]   * x[i] for i in imoveis_ids]) <= budget, "Restricao_Orcamento"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds, gapRel=gap_rel)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    print(f"  Status da otimizacao: {status}")

    # Extrair solução
    imoveis_selecionados = [i for i in imoveis_ids if x[i].varValue == 1.0]
    custo_total           = sum(custos[i] for i in imoveis_selecionados)
    retorno_total         = sum(retornos[i] for i in imoveis_selecionados)

    print(f"  Imoveis selecionados: {len(imoveis_selecionados)}")
    print(f"  Investimento total:   R$ {custo_total:,.2f} (orcamento: R$ {budget:,.2f})")
    print(f"  Retorno anual esperado: R$ {retorno_total:,.2f}")

    # Marcar imóveis selecionados (permite salvar todos os candidatos num único CSV)
    df_opt['Selecionado'] = df_opt.index.isin(imoveis_selecionados)

    portfolio_cols = [
        c for c in ['url', 'Bairro', sale_col,
                    'Receita Anual p5', 'Receita Anual Media', 'Receita Anual p95',
                    'Retorno Esperado Ajustado', 'VPL Estimado']
        if c in df_opt.columns
    ]

    return {
        'status': status,
        'imoveis_selecionados': imoveis_selecionados,
        'custo_total': custo_total,
        'retorno_total_esperado': retorno_total,
        'df_portfolio': df_opt.loc[imoveis_selecionados, portfolio_cols] if imoveis_selecionados else pd.DataFrame(),
        'df_opt': df_opt,
        'macro_rates': macro_rates,
    }
