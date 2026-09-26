"""
features de cada revisão, calculadas só com o que já existe quando a OS fecha.

o notebook treina com esta função e o inferencia.py classifica com a mesma:
se as duas contas divergirem, o modelo recebe em produção uma coisa
diferente do que aprendeu.

entrada: uma linha por visita (MaintenanceID já agregado), com as colunas
    vin, data, abertura, fechamento, revisao, km, concessionaria, fonte,
    agendado, codigo_servico, modelo, ano_modelo, data_nf, data_venda,
    data_entrega, data_emplacamento, data_garantia
opcional: volume_loja_12m (em produção vem do core, que tem o histórico da loja)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KM_PLANO = 10_000          # revisão a cada 10 mil km...
DIAS_PLANO = 365           # ...ou 12 meses, o que vier primeiro
GARANTIA_DIAS = 3 * 365

VEICULO = ["modelo", "defasagem_ano_modelo", "dias_estoque", "lag_entrega", "lag_emplacamento", "mes_venda"]
VISITA = ["revisao", "km", "idade_dias", "km_dia", "agendado", "fonte", "codigo_servico",
          "duracao_os", "mes_visita"]
PLANO = ["atraso_plano_dias", "km_excedente", "dias_ate_proxima", "proxima_fora_garantia",
         "dias_restantes_garantia"]
HISTORICO = ["n_visitas_anteriores", "intervalo_dias", "km_desde_anterior", "revisoes_puladas",
             "trocou_concessionaria", "n_concessionarias", "atraso_medio_hist"]
LOJA = ["concessionaria", "volume_loja_12m"]
GRUPOS = {"veiculo": VEICULO, "visita": VISITA, "plano": PLANO, "historico": HISTORICO, "loja": LOJA}
CANDIDATAS = VEICULO + VISITA + PLANO + HISTORICO + LOJA


def _dias_validos(delta: pd.Series, minimo: int, maximo: int) -> pd.Series:
    dias = delta.dt.days
    return dias.where(dias.between(minimo, maximo))


def volume_loja_12m(v: pd.DataFrame, inicio: pd.Timestamp | None = None) -> pd.Series:
    """visitas da loja nos 365 dias anteriores (o próprio dia fica de fora)."""
    inicio = inicio if inicio is not None else v["data"].min()
    tmp = v[["concessionaria", "data"]].sort_values(["concessionaria", "data"], kind="stable")
    datas = tmp["data"].to_numpy()
    contagem = np.zeros(len(tmp))
    # rolling(on=...) indexa pela data e perde a linha; searchsorted por loja não tem essa pegadinha
    for pos in tmp.groupby("concessionaria", sort=False).indices.values():
        d = datas[pos]
        contagem[pos] = np.searchsorted(d, d, side="left") - np.searchsorted(d, d - np.timedelta64(365, "D"), side="left")
    janela = pd.Series(contagem, index=tmp.index)
    # no começo da base a janela ainda não tem 12 meses: anualiza pra não punir 2020
    historico = (v["data"] - inicio).dt.days.clip(30, 365)
    return (janela.reindex(v.index) * 365 / historico).round(1)


def montar_features(visitas: pd.DataFrame, inicio_base: pd.Timestamp | None = None) -> pd.DataFrame:
    v = visitas.sort_values(["vin", "data"], kind="stable")
    g = v.groupby("vin", sort=False)
    ant_data = g["data"].shift(1)
    ant_km = g["km"].shift(1)
    ant_rev = g["revisao"].shift(1)
    ant_loja = g["concessionaria"].shift(1)
    idade = (v["data"] - v["data_garantia"]).dt.days.clip(lower=0)

    # mesmas regras da limpeza do notebook: em produção o histórico chega cru do core
    km = v["km"].where(v["km"].between(100, 500_000))
    km = km.where(km / idade.clip(lower=30) <= 1000)
    km = km.where(~(km < km.groupby(v["vin"], sort=False).cummax().groupby(v["vin"], sort=False).shift(1)))

    out = pd.DataFrame(index=v.index)

    # veículo: tudo isso já existe no dia da venda
    out["modelo"] = v["modelo"].astype(str)
    out["defasagem_ano_modelo"] = (pd.to_numeric(v["ano_modelo"]) - v["data_venda"].dt.year).clip(-2, 2)
    # nf da fábrica pra loja até a venda: carro encalhado costuma sair com desconto
    out["dias_estoque"] = _dias_validos(v["data_venda"] - v["data_nf"], 0, 730)
    out["lag_entrega"] = _dias_validos(v["data_entrega"] - v["data_venda"], 0, 365)
    out["lag_emplacamento"] = _dias_validos(v["data_emplacamento"] - v["data_entrega"], -30, 365)
    out["mes_venda"] = v["data_venda"].dt.month

    # visita atual
    km_dia = km / idade.clip(lower=30)
    out["revisao"] = v["revisao"].astype(float)
    out["km"] = km
    out["idade_dias"] = idade
    out["km_dia"] = km_dia
    out["agendado"] = v["agendado"].astype(int)
    out["fonte"] = v["fonte"].astype(str)
    out["codigo_servico"] = v["codigo_servico"].astype(str)
    out["duracao_os"] = _dias_validos(v["fechamento"] - v["abertura"], 0, 90)
    out["mes_visita"] = v["data"].dt.month

    # posição no plano de revisões
    dias_ate = np.minimum(DIAS_PLANO, KM_PLANO / km_dia.where(km_dia > 0))
    out["atraso_plano_dias"] = idade - DIAS_PLANO * v["revisao"]
    out["km_excedente"] = km - KM_PLANO * v["revisao"]
    out["dias_ate_proxima"] = dias_ate
    # a próxima revisão cai depois da garantia acabar: é aqui que o cliente sai
    out["proxima_fora_garantia"] = ((idade + dias_ate.fillna(DIAS_PLANO)) > GARANTIA_DIAS).astype(int)
    out["dias_restantes_garantia"] = (GARANTIA_DIAS - idade).clip(-730, GARANTIA_DIAS)

    # histórico do VIN até esta visita (shift/cumsum: nunca olha pra frente)
    primeira = g.cumcount() == 0
    out["n_visitas_anteriores"] = g.cumcount()
    out["intervalo_dias"] = (v["data"] - ant_data).dt.days.where(~primeira, idade)
    desde_ant = km - ant_km
    out["km_desde_anterior"] = desde_ant.where(desde_ant >= 0).where(~primeira, km)
    salto = (v["revisao"] - ant_rev - 1).clip(lower=0).fillna((v["revisao"] - 1).clip(lower=0))
    out["revisoes_puladas"] = salto.groupby(v["vin"], sort=False).cumsum()
    out["trocou_concessionaria"] = ((v["concessionaria"] != ant_loja) & ant_loja.notna()).astype(int)
    primeira_na_loja = ~v.duplicated(["vin", "concessionaria"])
    out["n_concessionarias"] = primeira_na_loja.astype(int).groupby(v["vin"], sort=False).cumsum()
    out["atraso_medio_hist"] = (out["atraso_plano_dias"].groupby(v["vin"], sort=False).cumsum()
                                / (out["n_visitas_anteriores"] + 1))

    # concessionária
    out["concessionaria"] = v["concessionaria"].astype(str)
    if "volume_loja_12m" in v:
        out["volume_loja_12m"] = v["volume_loja_12m"].astype(float)
    else:
        out["volume_loja_12m"] = volume_loja_12m(v, inicio_base)

    return out[CANDIDATAS].reindex(visitas.index)
