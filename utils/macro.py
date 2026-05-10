"""
utils/macro.py
--------------
Módulo de captura de variáveis macroeconômicas para o modelo de otimização.

Conteúdo:
    - MacroRates: dataclass com as taxas macroeconômicas e parâmetros derivados.
    - fetch_macro_rates: tenta buscar projeções via API do Banco Central (SGS/python-bcb).
                        Em caso de falha, retorna os valores de fallback hard-coded.

APIs consultadas (por ordem de tentativa):
    1. python-bcb  → séries SGS do Banco Central do Brasil
       - Série 432:  Taxa Selic Over (% a.a.)
       - Série 13522: Expectativa de IPCA acumulado 12 meses (Focus/BCB)
    2. Se python-bcb não estiver disponível: usa apenas os valores de fallback.

Referências:
    - SGS BCB: https://www3.bcb.gov.br/sgspub/
    - Focus/Expectativas: https://www.bcb.gov.br/publicacoes/focus
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ==============================================================================
# FALLBACK — Valores hard-coded (cenário base projetado para os próximos 12 meses)
# Fonte de referência: Relatório Focus BCB — atualizar conforme o relatório mais recente.
# Última atualização dos valores: Mai/2025
# ==============================================================================

#: Taxa Selic projetada acumulada em 12 meses (% ao ano, em decimal).
#: Fonte: mediana Focus/BCB para 2025/2026.
FALLBACK_SELIC_ANUAL: float = 0.1425          # 14,25 % a.a.

#: IPCA projetado acumulado em 12 meses (% ao ano, em decimal).
#: Fonte: mediana Focus/BCB para 2025/2026.
FALLBACK_IPCA_ANUAL: float = 0.0510           # 5,10 % a.a.

#: Spread de risco imobiliário acima da taxa livre de risco (% ao ano, em decimal).
#: Representa o prêmio exigido sobre a Selic por investir em imóveis vs. renda fixa.
#: Referência: diferencial histórico entre CRI/FII e Selic (~2,0 a 3,5 p.p.).
FALLBACK_SPREAD_IMOVEL: float = 0.025         # 2,50 p.p. a.a.

#: Horizonte de capitalização (meses) — período de análise do investimento.
#: 12 meses = análise anual (alinhado ao `simulate_annual_revenue`).
FALLBACK_HORIZONTE_MESES: int = 12


# ==============================================================================
# DATACLASS DE TAXAS
# ==============================================================================

@dataclass
class MacroRates:
    """
    Contêiner imutável com as taxas macroeconômicas e parâmetros financeiros
    derivados, prontos para consumo pelo modelo de otimização.

    Attributes:
        selic_anual (float): Taxa Selic projetada (decimal, ex: 0.1425 = 14,25% a.a.).
        ipca_anual (float): IPCA projetado (decimal, ex: 0.051 = 5,1% a.a.).
        spread_imovel (float): Spread de risco imobiliário (decimal, ex: 0.025 = 2,5 p.p.).
        horizonte_meses (int): Número de meses do horizonte de análise.
        fonte (str): Indica se os dados vieram de API ou do fallback hard-coded.
        taxa_desconto_nominal (float): Selic + spread (taxa de desconto nominal total).
        taxa_desconto_real (float): Taxa real via equação de Fisher:
                                    [(1 + nominal) / (1 + IPCA)] - 1
        fator_vp (float): Fator de valor presente de anuidade mensal:
                          [1 - (1 + i_mes)^{-T}] / i_mes
                          Multiplica a receita anual para obter o VPL do horizonte.
    """
    selic_anual: float
    ipca_anual: float
    spread_imovel: float
    horizonte_meses: int
    fonte: str

    # Campos derivados — calculados no __post_init__
    taxa_desconto_nominal: float = field(init=False)
    taxa_desconto_real: float = field(init=False)
    fator_vp: float = field(init=False)

    def __post_init__(self) -> None:
        # Taxa de desconto nominal = Selic + spread imobiliário
        self.taxa_desconto_nominal = self.selic_anual + self.spread_imovel

        # Taxa real via equação de Fisher
        self.taxa_desconto_real = (
            (1 + self.taxa_desconto_nominal) / (1 + self.ipca_anual)
        ) - 1

        # Converter taxa anual para mensal (capitalização composta)
        i_mes = (1 + self.taxa_desconto_real) ** (1 / 12) - 1

        # Fator de valor presente de anuidade com pagamentos mensais
        if i_mes > 1e-9:
            self.fator_vp = (1 - (1 + i_mes) ** (-self.horizonte_meses)) / i_mes
        else:
            # Taxa próxima de zero: VP ≈ soma direta dos fluxos
            self.fator_vp = float(self.horizonte_meses)

    def __str__(self) -> str:
        return (
            f"MacroRates("
            f"Selic={self.selic_anual:.2%}, "
            f"IPCA={self.ipca_anual:.2%}, "
            f"Spread={self.spread_imovel:.2%}, "
            f"i_nominal={self.taxa_desconto_nominal:.2%}, "
            f"i_real={self.taxa_desconto_real:.2%}, "
            f"FatorVP={self.fator_vp:.4f}, "
            f"T={self.horizonte_meses}m, "
            f"fonte={self.fonte!r})"
        )


# ==============================================================================
# FUNÇÃO PRINCIPAL DE CAPTURA
# ==============================================================================

def fetch_macro_rates(
    horizonte_meses: int = FALLBACK_HORIZONTE_MESES,
    spread_imovel: float = FALLBACK_SPREAD_IMOVEL,
    janela_dias: int = 5,
) -> MacroRates:
    """
    Busca projeções de Selic e IPCA para os próximos 12 meses.

    Estratégia (tentativa em ordem):
        1. python-bcb (SGS do Banco Central) — série 432 (Selic) e 13522 (IPCA Focus).
        2. Valores de fallback hard-coded (veja as constantes FALLBACK_* acima).

    Em caso de erro de rede, timeout ou ausência do pacote `python-bcb`,
    a função retorna silenciosamente os valores de fallback e registra um
    aviso via `logging.warning`.

    Args:
        horizonte_meses (int): Horizonte de capitalização em meses (padrão: 12).
        spread_imovel (float): Spread de risco imobiliário a ser somado à Selic
                               para obter a taxa de desconto nominal.
        janela_dias (int): Quantos dias anteriores à data atual buscar nas séries
                           SGS para obter o valor mais recente disponível.

    Returns:
        MacroRates: Objeto com todas as taxas e parâmetros derivados preenchidos.

    Example:
        >>> rates = fetch_macro_rates()
        >>> print(rates)
        MacroRates(Selic=14.25%, IPCA=5.10%, ...)
        >>> # Usar o fator de VP na otimização:
        >>> vpl_imovel = receita_anual * rates.fator_vp
    """
    selic = _fetch_bcb(horizonte_meses=horizonte_meses, janela_dias=janela_dias)

    if selic is not None:
        selic_val, ipca_val, fonte = selic
        return MacroRates(
            selic_anual=selic_val,
            ipca_anual=ipca_val,
            spread_imovel=spread_imovel,
            horizonte_meses=horizonte_meses,
            fonte=fonte,
        )

    # Fallback
    logger.warning(
        "Usando taxas macro hard-coded (fallback). "
        "Verifique a conectividade com o Banco Central (python-bcb / SGS)."
    )
    return MacroRates(
        selic_anual=FALLBACK_SELIC_ANUAL,
        ipca_anual=FALLBACK_IPCA_ANUAL,
        spread_imovel=spread_imovel,
        horizonte_meses=horizonte_meses,
        fonte="fallback_hardcoded",
    )


# ==============================================================================
# HELPERS INTERNOS
# ==============================================================================

def _fetch_bcb(
    horizonte_meses: int,
    janela_dias: int,
) -> Optional[tuple[float, float, str]]:
    """
    Tenta obter Selic e IPCA via python-bcb (SGS).

    Returns:
        tuple (selic_anual, ipca_anual, fonte) se bem-sucedido, None caso contrário.

    Séries SGS utilizadas:
        - 432:   Taxa Selic Over (% ao dia, anualizada pelo BCB como % ao ano)
        - 13522: IPCA acumulado esperado nos próximos 12 meses (mediana Focus)
    """
    try:
        from bcb import sgs  # python-bcb
    except ImportError:
        logger.warning(
            "Pacote 'python-bcb' não encontrado. "
            "Instale com: pip install python-bcb"
        )
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Série 432: Selic Over (% a.a.)
            df_selic = sgs.get({'selic': 432}, last=1)
            # Série 13522: IPCA acumulado 12m — mediana Focus
            df_ipca  = sgs.get({'ipca_focus': 13522}, last=1)

        if df_selic.empty or df_ipca.empty:
            logger.warning("SGS retornou série(s) vazia(s). Usando fallback.")
            return None

        selic_pct = float(df_selic['selic'].dropna().iloc[-1])   # % a.a.
        ipca_pct  = float(df_ipca['ipca_focus'].dropna().iloc[-1])  # % a.a.

        selic_dec = selic_pct / 100.0
        ipca_dec  = ipca_pct  / 100.0

        logger.info(
            "Taxas obtidas via BCB/SGS — Selic: %.2f%% a.a. | IPCA Focus: %.2f%% a.a.",
            selic_pct, ipca_pct,
        )
        return selic_dec, ipca_dec, "bcb_sgs"

    except Exception as exc:  # noqa: BLE001
        logger.warning("Falha ao consultar BCB/SGS: %s. Usando fallback.", exc)
        return None
