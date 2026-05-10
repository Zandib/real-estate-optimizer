"""
app.py — Dashboard de Otimização de Portfólio Imobiliário
==========================================================
Entrypoint Streamlit.  Execute:
    streamlit run app.py

A lógica de negócio fica nos módulos utils/*.  Este arquivo
apenas monta a UI, chama as funções e exibe os resultados.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import cloudpickle as pickle

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
PATH_MOD  = os.path.join(ROOT, "models")
PATH_DATA = os.path.join(ROOT, "data", "processed")
sys.path.insert(0, ROOT)

from utils.optimization import simulate_annual_revenue, optimize_portfolio
from utils.modeling      import create_target_derived_features
from utils.macro         import MacroRates, fetch_macro_rates

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Optimizer",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; color: #c9d1d9; }
  [data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
  .metric-card {
      background: #161b22; border: 1px solid #30363d; border-radius: 10px;
      padding: 18px 22px; text-align: center;
  }
  .metric-card .label { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
  .metric-card .value { color: #c9d1d9; font-size: 28px; font-weight: 700; margin-top: 4px; }
  .metric-card .delta { font-size: 13px; margin-top: 2px; }
  .pos { color: #3fb950; } .neg { color: #f85149; } .neu { color: #8b949e; }
  h1, h2, h3 { color: #c9d1d9 !important; }
  .stButton > button {
      background: linear-gradient(135deg, #2a9d8f, #264653);
      color: white; font-weight: 700; border: none;
      border-radius: 8px; padding: 12px 32px; width: 100%; font-size: 16px;
  }
  .stButton > button:hover { opacity: .85; }
  .fonte-badge-fallback {
      background: rgba(210,153,34,.18); color: #d29922;
      border-radius: 6px; padding: 3px 10px; font-size: 12px; font-weight: 700;
  }
  .fonte-badge-api {
      background: rgba(63,185,80,.18); color: #3fb950;
      border-radius: 6px; padding: 3px 10px; font-size: 12px; font-weight: 700;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHE: carrega artefatos pesados apenas uma vez por sessão
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Carregando modelos…")
def load_models():
    def _pkl(name):
        with open(os.path.join(PATH_MOD, name), "rb") as f:
            return pickle.load(f)
    return {
        "pipeline":   _pkl("full_preprocessing_pipeline.pkl"),
        "price":      _pkl("best_xgb_model.pkl"),
        "disp":       _pkl("best_xgb_model_disp.pkl"),
    }


@st.cache_data(show_spinner="Carregando base QuintoAndar…")
def load_base():
    return pd.read_csv(
        os.path.join(PATH_DATA, "base_features_quintoAndar.csv"), index_col=0
    )


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE DE PREDIÇÃO (preços + dias alugados)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Estimando aluguéis e ocupação…", ttl=3600)
def build_enriched_base(_pipeline, _price_model, _disp_model):
    """
    Aplica pipeline + modelos na base QuintoAndar.
    Cacheado com TTL = 1h pois não depende dos parâmetros do usuário.
    """
    df = load_base().copy()

    # Pré-processamento
    df_scaled = _pipeline.transform(df)
    feat_price = list(_price_model.feature_names_in_)
    df_proc    = df_scaled[feat_price]

    # Preço estimado
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["Aluguel estimado"] = np.expm1(_price_model.predict(df_proc))

    # Dias alugados estimados
    y_log = np.log1p(df["Aluguel estimado"])
    y_log.index = df_scaled.index
    derived = create_target_derived_features(df_scaled, y_log)
    X_disp  = pd.concat([df_scaled, derived], axis=1)[_disp_model.feature_names_in_]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dias = np.clip(_disp_model.predict(X_disp), 0, 365)
    df["Dias Alugados Estimado"] = dias
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — parâmetros do investidor
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏙️ Configurações")
    st.markdown("---")

    st.markdown("### 💰 Orçamento")
    orcamento = st.slider(
        "Orçamento total (R$)", 1_000_000, 30_000_000,
        value=10_000_000, step=500_000,
        format="R$ %d"
    )

    st.markdown("### ⚖️ Perfil de Risco")
    perfil = st.selectbox(
        "Perfil do investidor",
        ["Conservador", "Moderado", "Agressivo"],
        index=1,
    )
    _perfis = {
        "Conservador": (0.50, 0.35, 0.15),
        "Moderado":    (0.30, 0.50, 0.20),
        "Agressivo":   (0.15, 0.35, 0.50),
    }
    peso_p5, peso_media, peso_p95 = _perfis[perfil]
    with st.expander("Ajuste fino dos pesos"):
        peso_p5    = st.slider("Peso P5  (pessimista)",  0.0, 1.0, peso_p5,  0.05)
        peso_media = st.slider("Peso Média (moderado)",  0.0, 1.0, peso_media, 0.05)
        peso_p95   = st.slider("Peso P95 (otimista)",   0.0, 1.0, peso_p95,  0.05)
        soma = peso_p5 + peso_media + peso_p95
        if abs(soma - 1.0) > 0.01:
            st.warning(f"Pesos somam {soma:.2f} — devem somar 1.0")

    st.markdown("### 📈 Cenário Macroeconômico")
    usar_api = st.toggle("Buscar taxas do Banco Central", value=False)
    selic    = st.slider("Selic (% a.a.)",  5.0, 25.0, 14.25, 0.25, disabled=usar_api) / 100
    ipca     = st.slider("IPCA  (% a.a.)",  2.0, 15.0,  5.10, 0.10, disabled=usar_api) / 100
    spread   = st.slider("Spread imob. (p.p.)", 0.5, 5.0, 2.5, 0.25) / 100
    horizonte = st.slider("Horizonte (meses)", 6, 36, 12, 6)

    st.markdown("### 🎲 Monte Carlo")
    n_sim    = st.select_slider("Simulações por imóvel", [100, 500, 1000, 2000], value=1000)
    std_dias = st.slider("Incerteza de dias (std)", 10, 100, 50, 5)

    st.markdown("---")
    run_btn = st.button("🚀  Otimizar Portfólio")


# ══════════════════════════════════════════════════════════════════════════════
# CABEÇALHO
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🏙️ Real Estate Portfolio Optimizer")
st.markdown("Dashboard de decisão para alocação ótima de capital em imóveis.")
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO (sempre acontece, não depende do botão)
# ══════════════════════════════════════════════════════════════════════════════

models = load_models()
df_enriquecido = build_enriched_base(
    models["pipeline"], models["price"], models["disp"]
)

# ── card de taxas macroeconômicas ─────────────────────────────────────────────
if usar_api:
    macro = fetch_macro_rates(horizonte_meses=horizonte, spread_imovel=spread)
else:
    macro = MacroRates(
        selic_anual=selic, ipca_anual=ipca,
        spread_imovel=spread, horizonte_meses=horizonte,
        fonte="manual_slider",
    )
badge_cls  = "fonte-badge-api" if macro.fonte == "bcb_sgs" else "fonte-badge-fallback"
badge_txt  = "✓ API BCB" if macro.fonte == "bcb_sgs" else "⚠ Manual / Fallback"

st.markdown("### 📊 Cenário Macroeconômico Ativo")
cols_m = st.columns(7)
_taxa_labels = [
    ("Selic", f"{macro.selic_anual:.2%}"),
    ("IPCA",  f"{macro.ipca_anual:.2%}"),
    ("Spread", f"{macro.spread_imovel:.2%}"),
    ("i Nominal", f"{macro.taxa_desconto_nominal:.2%}"),
    ("i Real",    f"{macro.taxa_desconto_real:.2%}"),
    ("Fator VP",  f"{macro.fator_vp:.4f}"),
    ("Horizonte", f"{macro.horizonte_meses}m"),
]
for col, (lbl, val) in zip(cols_m, _taxa_labels):
    col.markdown(f"""
    <div class="metric-card">
      <div class="label">{lbl}</div>
      <div class="value" style="font-size:20px">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown(
    f'<span class="{badge_cls}">{badge_txt}</span>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# RESULTADO DA OTIMIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

if "resultado" not in st.session_state:
    st.info("Configure os parâmetros na barra lateral e clique em **🚀 Otimizar Portfólio**.")

if run_btn:
    if abs(peso_p5 + peso_media + peso_p95 - 1.0) > 0.01:
        st.error("Os pesos de risco devem somar 1.0. Ajuste-os antes de otimizar.")
        st.stop()

    with st.spinner("Executando Monte Carlo + solver MIP…"):
        df_sim = simulate_annual_revenue(
            df=df_enriquecido.copy(),
            daily_rent_col="Aluguel estimado",
            n_simulations=n_sim,
            mean_rented_days=df_enriquecido["Dias Alugados Estimado"].values,
            std_rented_days=std_dias,
            min_rented_days=0.0,
            max_rented_days=365.0,
            random_seed=42,
        )
        resultado = optimize_portfolio(
            df=df_sim,
            budget=orcamento,
            sale_col="Valor de venda",
            peso_p5=peso_p5,
            peso_media=peso_media,
            peso_p95=peso_p95,
            time_limit_seconds=60,
            gap_rel=0.01,
            macro_rates=macro,
        )
    st.session_state["resultado"] = resultado
    st.session_state["orcamento"] = orcamento
    st.session_state["macro"]     = macro

# ── exibir resultado ──────────────────────────────────────────────────────────
if "resultado" in st.session_state:
    res  = st.session_state["resultado"]
    orc  = st.session_state["orcamento"]
    mac  = st.session_state["macro"]
    df_p = res["df_portfolio"]

    custo     = res["custo_total"]
    retorno   = res["retorno_total_esperado"]
    n_imoveis = len(res["imoveis_selecionados"])
    caixa     = orc - custo
    yld_anual = (retorno / mac.fator_vp) / custo if custo > 0 else 0
    status    = res["status"]

    # ── métricas de topo ─────────────────────────────────────────────────────
    st.markdown("### 📌 Resumo do Portfólio Ótimo")
    c1, c2, c3, c4, c5 = st.columns(5)
    _status_color = "#3fb950" if status == "Optimal" else "#ffa657"

    def _card(col, label, value, delta="", delta_cls="neu"):
        col.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="delta {delta_cls}">{delta}</div>
        </div>""", unsafe_allow_html=True)

    _card(c1, "Status MIP",       status,
          delta_cls="pos" if status == "Optimal" else "neg")
    _card(c2, "Imóveis",          n_imoveis)
    _card(c3, "Investido",        f"R$ {custo/1e6:.2f}M",
          f"{custo/orc:.1%} do orçamento", "pos" if custo > 0 else "neg")
    _card(c4, "Retorno Esperado (VPL)", f"R$ {retorno/1e6:.2f}M",
          f"Yield anual: {yld_anual:.1%}", "pos" if yld_anual > mac.selic_anual else "neg")
    _card(c5, "Capital em Caixa", f"R$ {caixa/1e6:.2f}M",
          f"{caixa/orc:.1%} não investido", "neu")

    st.markdown("---")

    # ── tabela interativa ─────────────────────────────────────────────────────
    st.markdown("### 🏠 Imóveis Selecionados")
    if not df_p.empty:
        display_cols = [c for c in [
            "Bairro", "Valor de venda",
            "Receita Anual p5", "Receita Anual Media", "Receita Anual p95",
            "Retorno Esperado Ajustado", "VPL Estimado",
        ] if c in df_p.columns]
        df_show = df_p[display_cols].copy()
        df_show.attrs = {}  # Limpa attrs para evitar ValueError no pandas Styler

        # Yield por imóvel
        if "Retorno Esperado Ajustado" in df_show and "Valor de venda" in df_show:
            df_show["Yield Anual"] = df_show["Retorno Esperado Ajustado"] / df_show["Valor de venda"]

        # Formatar moeda
        fmt_r = {c: "{:,.0f}" for c in df_show.select_dtypes("number").columns
                 if c != "Yield Anual"}
        fmt_r["Yield Anual"] = "{:.1%}"
        st.dataframe(
            df_show.style.format(fmt_r),
            height=400,
        )
    else:
        st.warning("Nenhum imóvel selecionado. Tente aumentar o orçamento ou ajustar os pesos.")

    st.markdown("---")

    # ── gráficos ──────────────────────────────────────────────────────────────
    st.markdown("### 📈 Análise Visual")
    tab1, tab2, tab3 = st.tabs(["Distribuição de Retornos", "Yield vs Selic", "Alocação"])

    # TAB 1: box plot p5/média/p95 por imóvel
    with tab1:
        if not df_p.empty and {"Receita Anual p5","Receita Anual Media","Receita Anual p95"} <= set(df_p.columns):
            idx_label = df_p.get("Bairro", df_p.index.astype(str))
            fig1 = go.Figure()
            for col, name, color in [
                ("Receita Anual p5",   "P5 (pessimista)", "#e9c46a"),
                ("Receita Anual Media","Média",           "#2a9d8f"),
                ("Receita Anual p95",  "P95 (otimista)",  "#3fb950"),
            ]:
                if col in df_p.columns:
                    fig1.add_trace(go.Bar(
                        name=name, x=idx_label.values, y=df_p[col].values / 1e3,
                        marker_color=color,
                    ))
            fig1.update_layout(
                barmode="group", template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                title="Receita Anual por Imóvel — Cenários Monte Carlo (R$ mil)",
                yaxis_title="R$ mil / ano", xaxis_tickangle=-40,
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=60, b=100),
            )
            st.plotly_chart(fig1, width='stretch')

    # TAB 2: yield por imóvel vs Selic
    with tab2:
        if not df_p.empty and "Retorno Esperado Ajustado" in df_p.columns and "Valor de venda" in df_p.columns:
            yields = (df_p["Retorno Esperado Ajustado"] / df_p["Valor de venda"]).sort_values()
            labels = df_p.get("Bairro", df_p.index.astype(str)).loc[yields.index]
            colors = ["#3fb950" if v > mac.selic_anual else "#e9c46a" for v in yields]

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=yields.values * 100, y=labels.values,
                orientation="h", marker_color=colors,
                text=[f"{v:.1%}" for v in yields.values],
                textposition="outside",
            ))
            fig2.add_vline(
                x=mac.selic_anual * 100, line_dash="dash",
                line_color="#4361ee", annotation_text=f"Selic {mac.selic_anual:.2%}",
                annotation_font_color="#4361ee",
            )
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                title="Yield Anual por Imóvel vs. Selic (🟢 acima | 🟡 abaixo)",
                xaxis_title="Yield Bruto Anual (%)",
                margin=dict(l=160, r=80),
                height=max(350, 30 * len(yields)),
            )
            st.plotly_chart(fig2, width='stretch')

    # TAB 3: pizza de alocação
    with tab3:
        fig3 = go.Figure(go.Pie(
            labels=["Investido em imóveis", "Capital em caixa"],
            values=[custo, caixa],
            hole=0.55,
            marker_colors=["#2a9d8f", "#e9c46a"],
            textinfo="label+percent",
            textfont_size=13,
        ))
        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            title=f"Alocação do Orçamento — R$ {orc/1e6:.1f} MM",
            annotations=[dict(
                text=f"R$ {custo/1e6:.2f}M<br>investido",
                x=0.5, y=0.5, font_size=14, showarrow=False,
                font_color="#c9d1d9",
            )],
        )
        st.plotly_chart(fig3, width='stretch')

    # ── benchmarks ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Retorno do Portfólio vs. Benchmarks")
    selic_v  = mac.selic_anual
    ipca_v   = mac.ipca_anual
    yld_real = (1 + yld_anual) / (1 + ipca_v) - 1

    fig4 = go.Figure(go.Bar(
        x=["IPCA\n(inflação)", "Selic\n(benchmark)", "Yield Bruto\nPortfólio", "Yield Real\nPortfólio"],
        y=[ipca_v*100, selic_v*100, yld_anual*100, yld_real*100],
        marker_color=["#e9c46a", "#4361ee", "#2a9d8f", "#3fb950"],
        text=[f"{v:.1f}%" for v in [ipca_v*100, selic_v*100, yld_anual*100, yld_real*100]],
        textposition="outside", textfont=dict(color="white", size=14),
    ))
    fig4.add_hline(y=selic_v*100, line_dash="dash", line_color="#4361ee", opacity=0.5)
    fig4.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        yaxis_title="Taxa anual (%)",
        title="Benchmarks Macroeconômicos vs. Retorno do Portfólio",
        showlegend=False, margin=dict(t=60),
    )
    st.plotly_chart(fig4, width='stretch')

# ── rodapé ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8b949e;font-size:12px'>"
    "Real Estate Optimizer · FGV Trabalho Aplicado · "
    "Pipeline: XGBoost + Monte Carlo + PuLP MIP"
    "</div>",
    unsafe_allow_html=True,
)
