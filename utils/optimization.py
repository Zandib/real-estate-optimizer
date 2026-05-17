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
import gc

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

    # Liberação de memória pesada (matriz de N_imóveis x N_simulações)
    del revenue_matrix
    gc.collect()

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
            - 'custo_imoveis': custo total dos imóveis selecionados (R$)
            - 'v_selic_alocado': capital destinado à Selic (R$); quando macro_rates
                                 está ativo, equivale a budget − custo_imoveis.
            - 'retorno_imoveis': soma dos coeficientes VPL dos imóveis selecionados
            - 'retorno_selic': retorno anual projetado da parcela Selic (R$)
            - 'retorno_total_esperado': retorno_imoveis + retorno_selic
            - 'df_portfolio': DataFrame filtrado com os imóveis selecionados
            - 'df_opt': None (liberado para poupar memória)
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
    # Coeficiente da Função Objetivo — Homogeneização de Unidades
    # ------------------------------------------------------------------
    # PROBLEMA ANTERIOR: VPL[i] = Retorno_Ajustado[i] × fator_vp
    #   Para 12 meses a 11% real: fator_vp ≈ 11,3 (anuidade mensal)
    #   Isso amplia o retorno do imóvel em ~11x, enquanto a Selic usa
    #   selic_anual ≈ 0,1425 — bases completamente diferentes.
    #   Resultado: imóveis sempre vencem, independente do yield real.
    #
    # CORREÇÃO: ambos os ativos competem na mesma base anual (R$/ano).
    #   Imóvel i vence a Selic se e somente se:
    #       Retorno_Ajustado[i] / Custo[i]  >  selic_anual
    #   O VPL continua sendo calculado como KPI de exibição.
    # ------------------------------------------------------------------
    if macro_rates is not None:
        # VPL para exibição e KPI — não entra na FO (evita mismatch)
        df_opt['VPL Estimado'] = df_opt['Retorno Esperado Ajustado'] * macro_rates.fator_vp

        # Diagnóstico: quantos imóveis superam a Selic?
        yields_prop  = df_opt['Retorno Esperado Ajustado'] / df_opt[sale_col]
        n_acima      = (yields_prop > macro_rates.selic_anual).sum()
        print(f"  Custo de oportunidade: Selic={macro_rates.selic_anual:.2%} a.a.")
        print(f"  Imóveis com yield > Selic: {n_acima}/{len(df_opt)}")
        print(f"  Yield médio candidatos:    {yields_prop.mean():.2%} a.a.")

        coeficiente_objetivo = 'Retorno Esperado Ajustado'
        modo_objetivo = (
            f"Retorno Anual Bruto vs. Selic ({macro_rates.selic_anual:.2%} a.a.) "
            f"| VPL exib.: FV={macro_rates.fator_vp:.4f}"
        )
    else:
        coeficiente_objetivo = 'Retorno Esperado Ajustado'
        modo_objetivo        = "Retorno Bruto Anual (sem desconto)"

    imoveis_ids = df_opt.index.tolist()
    custos   = df_opt[sale_col].to_dict()
    retornos = df_opt[coeficiente_objetivo].to_dict()

    print(f"  Iniciando otimizacao MIP para {len(imoveis_ids)} imoveis candidatos...")
    print(f"  Modo objetivo: {modo_objetivo}")

    # ------------------------------------------------------------------
    # Ativo livre de risco: Selic como alternativa de alocação
    # ------------------------------------------------------------------
    # Quando macro_rates está ativo, v_selic é uma variável contínua que
    # representa o montante alocado em renda fixa (Selic). Como seu
    # coeficiente na FO (selic_anual) é positivo, o solver sempre a
    # satura ao máximo permitido pela restrição orçamentária — garantindo
    # que 100% do capital seja alocado: v_selic = budget − Σcusto·x.
    # O modelo é livre para colocar 0% em imóveis se a Selic superar
    # o VPL de todos os candidatos.
    # ------------------------------------------------------------------
    use_selic_asset = macro_rates is not None

    prob = pulp.LpProblem("Otimizacao_Portfolio_Imoveis", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("comprar", imoveis_ids, cat='Binary')

    if use_selic_asset:
        selic_coef = macro_rates.selic_anual   # retorno anual por R$ na Selic
        v_selic    = pulp.LpVariable("v_selic", lowBound=0, cat="Continuous")

        # FO: max Σ VPL[i]·x[i] + v_selic · selic_anual
        prob += (
            pulp.lpSum([retornos[i] * x[i] for i in imoveis_ids])
            + v_selic * selic_coef
        ), "Max_Retorno_Total"

        # Restrição: imóveis + Selic ≤ orçamento
        prob += (
            pulp.lpSum([custos[i] * x[i] for i in imoveis_ids]) + v_selic <= budget
        ), "Restricao_Orcamento"
    else:
        v_selic = None
        prob += pulp.lpSum([retornos[i] * x[i] for i in imoveis_ids]), "Max_Retorno_Objetivo"
        prob += pulp.lpSum([custos[i]   * x[i] for i in imoveis_ids]) <= budget, "Restricao_Orcamento"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds, gapRel=gap_rel)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    print(f"  Status da otimizacao: {status}")

    # ── Extrair solução imobiliária ────────────────────────────────────
    imoveis_selecionados = [i for i in imoveis_ids if x[i].varValue == 1.0]
    custo_imoveis   = sum(custos[i]   for i in imoveis_selecionados)
    retorno_imoveis = sum(retornos[i] for i in imoveis_selecionados)

    # ── Extrair alocação Selic ─────────────────────────────────────────
    # Todo capital não alocado em imóveis é considerado investido na Selic.
    if use_selic_asset:
        v_selic_alocado = float(v_selic.varValue or 0.0)
        retorno_selic   = v_selic_alocado * selic_coef
    else:
        v_selic_alocado = budget - custo_imoveis  # residual (sem rentabilização)
        retorno_selic   = 0.0

    retorno_total = retorno_imoveis + retorno_selic

    print(f"  Imoveis selecionados: {len(imoveis_selecionados)}")
    print(f"  Custo imoveis:        R$ {custo_imoveis:,.2f}")
    print(f"  Alocado na Selic:     R$ {v_selic_alocado:,.2f}")
    print(f"  Retorno imoveis:      R$ {retorno_imoveis:,.2f}")
    print(f"  Retorno Selic:        R$ {retorno_selic:,.2f}")
    print(f"  Retorno TOTAL:        R$ {retorno_total:,.2f}  |  orcamento: R$ {budget:,.2f}")

    df_opt['Selecionado'] = df_opt.index.isin(imoveis_selecionados)

    portfolio_cols = [
        c for c in ['url', 'Bairro', sale_col,
                    'Receita Anual p5', 'Receita Anual Media', 'Receita Anual p95',
                    'Receita Anual Std',
                    'Retorno Esperado Ajustado', 'VPL Estimado']
        if c in df_opt.columns
    ]

    return {
        'status':                 status,
        'imoveis_selecionados':   imoveis_selecionados,
        'custo_imoveis':          custo_imoveis,
        'custo_total':            custo_imoveis,          # alias retrocompat.
        'v_selic_alocado':        v_selic_alocado,
        'retorno_imoveis':        retorno_imoveis,
        'retorno_selic':          retorno_selic,
        'retorno_total_esperado': retorno_total,
        'df_portfolio': df_opt.loc[imoveis_selecionados, portfolio_cols] if imoveis_selecionados else pd.DataFrame(),
        'df_opt':        None,
        'macro_rates':   macro_rates,
    }
