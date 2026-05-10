<div align="center">
  <h1 align="center">🏙️ Real Estate Optimizer</h1>
  <p align="center">
    <strong>Sistema inteligente para otimização e alocação de portfólios imobiliários com Machine Learning e Pesquisa Operacional.</strong>
  </p>
  
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/PuLP-Optimization-4CAF50?style=for-the-badge" alt="PuLP">
    <img src="https://img.shields.io/badge/XGBoost-Machine_Learning-F37626?style=for-the-badge" alt="XGBoost">
  </p>
</div>

---

## 📌 Visão Geral

O **Real Estate Optimizer** é um ecossistema projetado para auxiliar investidores imobiliários e fundos a maximizar o **Valor Presente Líquido (VPL)** de seus portfólios sob estritas restrições orçamentárias (ex: teto de R$ 10 Milhões) e perfis de risco (Conservador, Moderado, Agressivo).

O pipeline funde previsões de aluguel e vacância usando **Machine Learning (XGBoost)**, mapeia a incerteza estatística através de **Simulações de Monte Carlo**, atualiza taxas macroeconômicas na **API do Banco Central do Brasil (BCB)** e toma decisões precisas de aquisição binária utilizando **Programação Inteira Mista (MIP)**.

## 🗂️ Arquitetura do Repositório

O projeto adota uma estrita separação de preocupações entre Interface (Frontend), Lógica de Negócios e Dados.

```text
real-estate-optimizer/
├── app.py                   # Frontend Streamlit (Interface interativa e relatórios)
├── data/
│   └── processed/           # Bases de dados otimizadas (Downcasted features)
├── models/                  # Pipelines serializados e modelos XGBoost em pickle
├── utils/                   # Módulos de Core Business Logic
│   ├── optimization.py      # Solver MIP (PuLP) e simulação de Monte Carlo
│   ├── modeling.py          # Regras preditivas de negócios e tratamentos de features alvo
│   ├── macro.py             # Integração com API SGS do BCB para IPCA e Selic
│   ├── geospatial.py        # Lógica de cálculo de distâncias (praias, metrô, escolas)
│   ├── data_cleaning.py     # Pipeline de sanitização de dados nulos e outliers
│   ├── quadtree.py          # Particionamento espacial e indexação geográfica rápida
│   └── visualization.py     # Auxiliares nativos de plotagem e gráficos
├── README.md                # Documentação técnica (Você está aqui)
└── requirements.txt         # Dependências do projeto (Pip)
```

## 🧮 Matemática do Modelo (Seção Técnica)

O núcleo de tomada de decisão é fundamentado em Programação Linear/Inteira e Matemática Financeira. O algoritmo avalia $N$ imóveis candidatos e decide quais comprar.

### 1. Taxa de Desconto e Matemática Financeira
A rentabilidade de um imóvel não é isenta de inflação ou do custo de oportunidade. Utilizamos a [Equação de Fisher](https://pt.wikipedia.org/wiki/Equa%C3%A7%C3%A3o_de_Fisher) para extrair a **Taxa Real** de desconto:

$$ i_{\text{real}} = \left( \frac{1 + \text{Selic} + \text{Spread}}{1 + \text{IPCA}} \right) - 1 $$

Convertida para taxa equivalente mensal:

$$ i_{\text{mês}} = (1 + i_{\text{real}})^{\frac{1}{12}} - 1 $$

O fluxo de locação do imóvel é contínuo ao longo dos meses. Para trazer os $T$ meses ao Valor Presente (ex: horizonte $T=12$ meses), calculamos o fator de valor presente de uma anuidade padrão:

$$ \text{FATOR\_VP} = \frac{1 - (1 + i_{\text{mês}})^{-T}}{i_{\text{mês}}} $$

### 2. Função Objetivo e Risco Ajustado (Monte Carlo)
A partir das simulações estatísticas do tempo de locação e dos aluguéis previstos, o algoritmo extrai percentis de lucro de cada imóvel $i$: o pessimista ($p5$), a média estatística, e o otimista ($p95$). 

O **Retorno Ajustado** ($R_i$) reflete o perfil do investidor através da calibração de pesos ($w$):

$$ R_i = w_{\text{p5}} \cdot (p5_i) + w_{\text{média}} \cdot (\text{Média}_i) + w_{\text{p95}} \cdot (p95_i) $$
$$\text{Onde:} \quad w_{\text{p5}} + w_{\text{média}} + w_{\text{p95}} = 1 $$

Isso é descontado no tempo, gerando o **Valor Presente Líquido (VPL)** escalar para preservação de linearidade:

$$ VPL_i = R_i \times \text{FATOR\_VP} $$

O **Problema de Otimização Combinatória (Knapsack Problem Expandido)** busca maximizar a soma do VPL dos imóveis comprados:

$$ \max \sum_{i=1}^{N} (VPL_i \cdot x_i) $$

### 3. Restrições do Sistema
O orçamento total disponível para aquisição limitará quais e quantos imóveis podem compor a carteira.

$$ \sum_{i=1}^{N} (Custo_i \cdot x_i) \leq B $$

Sendo a variável de decisão ($x_i$) estritamente binária (Comprar ou não Comprar o imóvel):

$$ x_i \in \{0, 1\} \quad \forall i \in \{1, \dots, N\} $$

## 🛠️ Pipeline de Machine Learning e Inferência

Para garantir que a otimização seja baseada em estimativas sólidas, o sistema conta com pipelines pesados de regressão treinados separadamente.

### 1. Treinamento de Regressores (`python/run_training.py`)
Responsável por treinar e serializar os modelos de regressão na base do Airbnb. O processo engloba:
- **Alvo (Target):** Estimar o preço diário do aluguel (em log) e os dias alugados/ano.
- **Engenharia de Features:** Pipeline de pré-processamento customizado, conversão de dummies, StandardScaler, remoção iterativa por VIF e p-valor (no modelo OLS). Criação de features avançadas como *`Diferenca_preco_bairro`* cruzadas no target.
- **Modelos Treinados:** Regressão Linear (OLS), KNN Regressor e, destacadamente, **XGBoost**.
- **Otimização:** Tuning de hiperparâmetros com `GridSearchCV` e validação cruzada para garantir a melhor generalização e salvar os artefatos `best_xgb_model.pkl` e `best_xgb_model_disp.pkl`.

### 2. Simulação de Rentabilidade (`python/run_rentabilidade.py`)
Utiliza os modelos previamente treinados para extrapolar métricas de locação e encontrar o subconjunto perfeito de imóveis:
- **Inferência (Sanity Check):** Testa o XGBoost contra o próprio Airbnb para confirmar a coesão do modelo (MAE/R²).
- **Projeção em QuintoAndar:** O modelo prevê por quanto as casas à venda (QuintoAndar) seriam alugadas (Preço vs Disponibilidade).
- **Simulação de Risco & Otimização:** Invoca as lógicas do pacote `utils.optimization` — roda simulações de Monte Carlo nas casas projetadas e aciona o Solver MIP (`PuLP`) finalizando na seleção do **Portfolio Ótimo**.

## 🛠️ Documentação de Métodos Principais (Core Lógico)

- **`utils.optimization.simulate_annual_revenue(...)`**: 
  - **Parâmetros**: `df` de candidatos, `n_simulations`, `mean_rented_days`.
  - **Lógica**: Extrai o aluguel predito e aplica simulações massivas de Monte Carlo criando amostras de disponibilidade em uma distribuição gaussiana (truncada em 0 e 365).
  - **Saída**: Retorna o DF original acrescido das colunas probabilísticas (`Receita Anual p5`, `Media`, `p95`).
- **`utils.optimization.optimize_portfolio(...)`**:
  - **Parâmetros**: `df_sim`, budget limite (`budget`), e os pesos de risco, bem como as taxas `MacroRates`.
  - **Lógica**: Configura as variáveis lineares de decisão na biblioteca `PuLP` limitadas pelo teto orçamentário e aciona o solver CBC (Coin-or branch and cut).
  - **Saída**: Retorna um dicionário em conformidade com UI englobando `status` do modelo, custo de aquisição final, retorno esperado e a tabela fracionada de imóveis selecionados.
- **`app.py: load_base()`**: 
  - Submete o dataset cru a uma rigorosa rotina de **Otimização de Memória e Caching**. 
  - Reduz os tipos Float64 e Int64, converte textos repetitivos via método paramétrico de `.astype('category')`, e executa coletor de lixo `gc.collect()` para impedir travamento de servidor por *Out Of Memory* (OOM).

## 🚀 Instruções de Execução

Requisitos de Sistema: Python 3.10+ recomendado.

**1. Clone o repositório e acesse a pasta:**
```bash
git clone https://github.com/Zandib/real-estate-optimizer.git
cd real-estate-optimizer
```

**2. Instale as dependências contidas no requirements.txt:**
```bash
pip install -r requirements.txt
```

**3. Execute o servidor do Streamlit:**
```bash
streamlit run app.py
```
> Após iniciar, a interface local será exposta geralmente na porta `:8501`. Os hiperparâmetros (como tolerância de spread e horizonte em meses) poderão ser ajustados livremente pela barra lateral da UI interativa.

## 💻 Stack Técnica
- **Linguagem:** Python
- **Interface & Dashboard:** Streamlit, Plotly
- **Pesquisa Operacional (MIP):** PuLP Solver (CBC)
- **Engenharia de Dados & Machine Learning:** Pandas, Scikit-Learn, XGBoost, Numpy
- **Otimização Interna:** Caching explícito, TTL expirations e Garbage Collection manual (`import gc`) para estabilidade em nuvem.
