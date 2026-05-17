"""
utils/visualization.py
----------------------
Módulo de visualização do Real Estate Optimizer.

Contém:
    - plot_montecarlo_distribution: Curva de densidade interativa (Plotly) dos retornos
      simulados pelo Monte Carlo do portfólio — pronta para st.plotly_chart().
    - plot_asset_allocation: Painel duplo (Waterfall de alocação + Barra empilhada de
      retorno por classe de ativo) — substitui a pizza quando a Selic é ativo elegivel.
    - Funções EDA legadas baseadas em Matplotlib/Seaborn (análise exploratória offline).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, gaussian_kde


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY — Monte Carlo: Distribuição de Retornos do Portfólio
# ══════════════════════════════════════════════════════════════════════════════

def plot_montecarlo_distribution(
    df_portfolio: pd.DataFrame,
    p5_col: str  = "Receita Anual p5",
    med_col: str = "Receita Anual Media",
    p95_col: str = "Receita Anual p95",
    std_col: str = "Receita Anual Std",
    selic_anual: Optional[float] = None,
    budget: Optional[float] = None,
    kde_points: int = 512,
) -> go.Figure:
    """
    Gera uma curva de densidade interativa (PDF) dos retornos anuais
    agregados do portfólio selecionado, com linhas de referência estatísticas.

    Estratégia de reconstrução da distribuição do portfólio:
    ─────────────────────────────────────────────────────────
    Pelo Teorema Central do Limite, a soma de N variáveis independentes com
    médias μᵢ e desvios σᵢ converge para uma Normal com:

        μ_portfólio  = Σᵢ μᵢ                  (soma das médias)
        σ_portfólio  = √(Σᵢ σᵢ²)              (raiz da soma das variâncias)

    Esta abordagem usa diretamente as colunas 'Receita Anual Media' e
    'Receita Anual Std' — salvas pela simulate_annual_revenue — sem reter
    a matriz bruta (N_imóveis × N_simulações) na memória.

    Se 'Receita Anual Std' não estiver disponível, faz fallback para
    estimativa via intervalo P5→P95 (≈ 3,29σ para distribuição Normal).

    Linhas de referência:
        - VaR P5  (vermelho): percentil 5 — cenário pessimista
        - Mediana P50 (branco): percentil 50 — ponto central da distribuição
        - P95     (verde):  percentil 95 — cenário otimista
        - Selic   (azul):   benchmark de custo de oportunidade

    Args:
        df_portfolio (pd.DataFrame): DataFrame com os imóveis selecionados.
        p5_col (str): Coluna com percentil 5 (VaR pessimista).
        med_col (str): Coluna com a média dos retornos por imóvel.
        p95_col (str): Coluna com percentil 95 (cenário otimista).
        std_col (str): Coluna com o desvio padrão por imóvel (preferencial).
        selic_anual (float | None): Taxa Selic anual para linha benchmark.
        budget (float | None): Capital investido para cálculo de yield no hover.
        kde_points (int): Resolução da grade KDE (padrão: 512).

    Returns:
        go.Figure: Figura Plotly pronta para st.plotly_chart().
    """
    import gc

    # ── 1. Validar colunas mínimas ────────────────────────────────────────────
    required = [p5_col, med_col, p95_col]
    missing  = [c for c in required if c not in df_portfolio.columns]
    if missing or df_portfolio.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Distribuição Monte Carlo — sem dados disponíveis",
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            annotations=[dict(
                text="Rode a otimização para visualizar a distribuição.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16, color="#8b949e"),
            )],
        )
        return fig

    # ── 2. Parâmetros da distribuição do portfólio via convolução ─────────────
    # μ_portfólio = Σ μᵢ  (aditividade das médias)
    mu_portfolio = float(df_portfolio[med_col].sum())

    # σ_portfólio = √(Σ σᵢ²)  (aditividade das variâncias — independência)
    if std_col in df_portfolio.columns and df_portfolio[std_col].notna().any():
        sigma_portfolio = float(
            np.sqrt((df_portfolio[std_col].fillna(0) ** 2).sum())
        )
        metodo_sigma = "convolução (√Σσᵢ²)"
    else:
        # Fallback: estimar σ pelo intervalo P5→P95 (≈ 3,29σ para Normal)
        total_p5  = float(df_portfolio[p5_col].sum())
        total_p95 = float(df_portfolio[p95_col].sum())
        sigma_portfolio = (total_p95 - total_p5) / 3.29
        metodo_sigma = "fallback (P95-P5)/3.29σ"

    sigma_portfolio = max(sigma_portfolio, 1.0)  # guard: evitar σ=0

    # Percentis derivados da distribuição Normal do portfólio
    from scipy.stats import norm
    total_p5  = float(norm.ppf(0.05, loc=mu_portfolio, scale=sigma_portfolio))
    total_p50 = float(norm.ppf(0.50, loc=mu_portfolio, scale=sigma_portfolio))  # mediana = P50
    total_p95 = float(norm.ppf(0.95, loc=mu_portfolio, scale=sigma_portfolio))

    # ── 3. Grade KDE sobre a distribuição Normal do portfólio ─────────────────
    # Suporte: μ ± 4σ (cobre 99,994% da massa de probabilidade)
    x_min   = mu_portfolio - 4.0 * sigma_portfolio
    x_max   = mu_portfolio + 4.0 * sigma_portfolio
    x_range = np.linspace(x_min, x_max, kde_points)

    # PDF analítica da Normal
    y_pdf = norm.pdf(x_range, loc=mu_portfolio, scale=sigma_portfolio)

    # ── 4. Helpers de formatação ──────────────────────────────────────────────
    def _fmt(v: float) -> str:
        if abs(v) >= 1e6:
            return f"R$ {v/1e6:.2f} MM"
        return f"R$ {v/1e3:.1f} mil"

    def _yield_txt(v: float) -> str:
        return f" | Yield: {v/budget:.1%}" if (budget and budget > 0) else ""

    # ── 5. Figura ─────────────────────────────────────────────────────────────
    fig = go.Figure()

    # Trace único: PDF (área preenchida)
    fig.add_trace(
        go.Scatter(
            x=x_range / 1e6,
            y=y_pdf,
            mode="lines",
            fill="tozeroy",
            name="Densidade (PDF)",
            line=dict(color="#2a9d8f", width=2.5),
            fillcolor="rgba(42,157,143,0.15)",
            hovertemplate=(
                "Retorno: R$ %{x:.2f} MM<br>"
                "PDF: %{y:.5f}<extra></extra>"
            ),
        )
    )

    # ── 6. Linhas de referência estatísticas ──────────────────────────────────
    # Calcula probabilidade acumulada de cada referência para anotação
    def _cdf_at(v: float) -> float:
        return float(norm.cdf(v, loc=mu_portfolio, scale=sigma_portfolio))

    refs = [
        (total_p5,  "#f85149", f"VaR P5<br>{_fmt(total_p5)}{_yield_txt(total_p5)}",   "dot"),
        (total_p50, "#c9d1d9", f"Mediana P50<br>{_fmt(total_p50)}{_yield_txt(total_p50)}", "dash"),
        (total_p95, "#3fb950", f"P95<br>{_fmt(total_p95)}{_yield_txt(total_p95)}",      "dot"),
    ]

    for val, color, label, dash in refs:
        fig.add_vline(
            x=val / 1e6,
            line_dash=dash,
            line_color=color,
            line_width=1.8,
            annotation_text=label,
            annotation_position="top left",
            annotation_font_color=color,
            annotation_font_size=11,
            annotation_bgcolor="rgba(13,17,23,0.75)",
        )

    # Linha Selic — benchmark de custo de oportunidade
    if selic_anual is not None and budget is not None and budget > 0:
        retorno_selic = selic_anual * budget
        prob_selic    = _cdf_at(retorno_selic)
        fig.add_vline(
            x=retorno_selic / 1e6,
            line_dash="longdash",
            line_color="#4361ee",
            line_width=1.5,
            annotation_text=(
                f"Selic {selic_anual:.2%}<br>"
                f"{_fmt(retorno_selic)}<br>"
                f"P(superar): {1-prob_selic:.0%}"
            ),
            annotation_position="top right",
            annotation_font_color="#4361ee",
            annotation_font_size=11,
            annotation_bgcolor="rgba(13,17,23,0.75)",
        )

    # ── 7. Layout ─────────────────────────────────────────────────────────────
    subtitle = (
        f"Portfólio: {len(df_portfolio)} imóveis  |  "
        f"P50={_fmt(total_p50)}  σ={_fmt(sigma_portfolio)}  "
        f"[{metodo_sigma}]  |  "
        f"P5={_fmt(total_p5)}  P95={_fmt(total_p95)}"
    )
    fig.update_layout(
        title=dict(
            text=(
                "📊 Distribuição de Probabilidade dos Retornos — Monte Carlo<br>"
                f"<sup style='color:#8b949e;font-size:12px'>{subtitle}</sup>"
            ),
            font=dict(size=17, color="#c9d1d9"),
            x=0.0,
            xanchor="left",
        ),
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        xaxis=dict(
            title="Retorno Anual Agregado do Portfólio (R$ MM)",
            tickformat=".2f",
            gridcolor="#21262d",
            zerolinecolor="#30363d",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.04,
            xanchor="right",  x=1,
            font=dict(size=12),
        ),
        margin=dict(t=100, b=60, l=70, r=70),
        hovermode="x unified",
        height=460,
    )

    # Eixo Y — densidade
    fig.update_yaxes(
        title_text="Densidade de Probabilidade (PDF)",
        showgrid=True,
        gridcolor="#21262d",
        zerolinecolor="#30363d",
    )

    # ── 8. Liberar memória antes de retornar ──────────────────────────────────
    del x_range, y_pdf
    gc.collect()

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY — Alocação de Ativos: Waterfall + Retorno por Classe
# ══════════════════════════════════════════════════════════════════════════════

def plot_asset_allocation(
    budget: float,
    custo_imoveis: float,
    v_selic: float,
    retorno_imoveis: float,
    retorno_selic: float,
    selic_anual: float,
) -> go.Figure:
    """
    Painel duplo interativo que detalha a decisão de alocação do solver MIP.

    Subplots:
        Esquerda — Waterfall de Alocação:
            Decomposição visual do orçamento total em:
            Orçamento → Imóveis → Selic → Total (= orçamento)

        Direita — Retorno por Classe de Ativo:
            Barra empilhada horizontal comparando:
            - Retorno total do portfólio (Imóveis + Selic)
            - Benchmark: retorno hipotético se 100% fosse na Selic

    Args:
        budget (float): Orçamento total (R$).
        custo_imoveis (float): Capital alocado em imóveis pelo solver (R$).
        v_selic (float): Capital alocado na Selic pelo solver (R$).
        retorno_imoveis (float): Retorno esperado (VPL) dos imóveis selecionados (R$).
        retorno_selic (float): Retorno projetado da parcela Selic (R$).
        selic_anual (float): Taxa Selic anual (decimal, ex: 0.1425).

    Returns:
        go.Figure: Figura Plotly com subplots 1×2 pronta para st.plotly_chart().
    """
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.48, 0.52],
        subplot_titles=[
            "💰 Alocação do Orçamento (R$ MM)",
            "📈 Retorno por Classe de Ativo (R$ MM)",
        ],
        horizontal_spacing=0.10,
    )

    M = 1e6  # escala para MM

    # ─────────────────────────────────────────────────────────────────────────────
    # ESQUERDA: Waterfall de Alocação
    # ─────────────────────────────────────────────────────────────────────────────
    pct_imoveis = custo_imoveis / budget if budget > 0 else 0
    pct_selic   = v_selic       / budget if budget > 0 else 0

    fig.add_trace(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Orçamento", "Imóveis", "Selic", "Total Alocado"],
            y=[budget / M, -custo_imoveis / M, -v_selic / M, 0],
            text=[
                f"R$ {budget/M:.2f}MM",
                f"R$ {custo_imoveis/M:.2f}MM\n({pct_imoveis:.1%})",
                f"R$ {v_selic/M:.2f}MM\n({pct_selic:.1%})",
                f"R$ {(custo_imoveis+v_selic)/M:.2f}MM",
            ],
            textposition="outside",
            connector={"line": {"color": "#30363d", "width": 1.5}},
            decreasing={"marker": {"color": "#2a9d8f"}},   # imóveis = verde-azulado
            increasing={"marker": {"color": "#4361ee"}},   # selic = azul
            totals={"marker": {"color": "#3fb950"}},
            hovertemplate="%{x}: R$ %{y:.3f}MM<extra></extra>",
        ),
        row=1, col=1,
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # DIREITA: Retorno por classe de ativo — portfólio vs. benchmark 100% Selic
    # ─────────────────────────────────────────────────────────────────────────────
    retorno_benchmark = budget * selic_anual
    retorno_total     = retorno_imoveis + retorno_selic
    categorias        = ["Portfólio\nÓtimo", "Benchmark\n(100% Selic)"]

    # Parcela Imóveis
    fig.add_trace(
        go.Bar(
            name="Imóveis",
            x=[retorno_imoveis / M, 0],
            y=categorias,
            orientation="h",
            marker_color="#2a9d8f",
            text=[f"Imóveis\nR$ {retorno_imoveis/M:.3f}MM", ""],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="Imóveis: R$ %{x:.3f} MM<extra></extra>",
        ),
        row=1, col=2,
    )

    # Parcela Selic (portfólio)
    fig.add_trace(
        go.Bar(
            name=f"Selic ({selic_anual:.2%} a.a.)",
            x=[retorno_selic / M, retorno_benchmark / M],
            y=categorias,
            orientation="h",
            marker_color="#4361ee",
            text=[
                f"Selic\nR$ {retorno_selic/M:.3f}MM",
                f"100% Selic\nR$ {retorno_benchmark/M:.3f}MM",
            ],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="Selic: R$ %{x:.3f} MM<extra></extra>",
        ),
        row=1, col=2,
    )

    # Linha de retorno total do portfólio
    fig.add_vline(
        x=retorno_total / M,
        line_dash="dash",
        line_color="#3fb950",
        line_width=1.8,
        annotation_text=f"Total portfólio\nR$ {retorno_total/M:.3f}MM",
        annotation_font_color="#3fb950",
        annotation_font_size=11,
        annotation_position="top right",
        row=1, col=2,
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Layout global
    # ─────────────────────────────────────────────────────────────────────────────
    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.08,
            xanchor="right",  x=1,
            font=dict(size=12),
        ),
        margin=dict(t=80, b=40, l=60, r=60),
        title=dict(
            text=(
                "🏦 Decisão de Alocação do Solver — Imóveis vs. Renda Fixa (Selic)<br>"
                f"<sup style='color:#8b949e;font-size:12px'>"
                f"Orçamento: R$ {budget/M:.1f}MM  |  "
                f"Imóveis: {custo_imoveis/budget:.1%}  |  "
                f"Selic: {v_selic/budget:.1%}  |  "
                f"Selic anual: {selic_anual:.2%}"
                "</sup>"
            ),
            font=dict(size=16, color="#c9d1d9"),
            x=0.0, xanchor="left",
        ),
    )
    fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#30363d")
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#30363d")

    return fig




def plot_singleVar_numeric(df, col, xlim=None, ylim=None):
    plt.figure(figsize=(18, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df[col].dropna(), kde=True, bins=50)
    plt.title(f'Distribuição: {col}')
    plt.xlabel(col)
    plt.ylabel('Frequencia')

    # Box plot
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

def plot_against_target_numerical(df, col, target):
    plt.figure(figsize=(18, 6))
    sns.scatterplot(x=col, y=target, data=df, alpha=0.6)

    # Calculate correlation and p-value
    df_cleaned = df[[col, target]].dropna()
    if not df_cleaned.empty:
        correlation, p_value = pearsonr(df_cleaned[col], df_cleaned[target])
        annotation_text = f'Correlação (Pearson): {correlation:.2f}\nP-valor: {p_value:.3f}'
        plt.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.9))
    else:
        plt.annotate('Não foi possível calcular a correlação', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.5))

    plt.title(f'{target} vs {col}')
    plt.xlabel(col)
    plt.ylabel(target)
    plt.tight_layout()
    plt.show()
