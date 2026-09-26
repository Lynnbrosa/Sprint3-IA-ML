"""
classifica o risco de evasão de um veículo a partir do histórico dele na rede.
é o que o ml-api do PrevioPLS roda quando uma OS de revisão fecha.

uso:
    python src/inferencia.py exemplo_requisicao.json

formato da requisição (datas em aaaa-mm-dd):
    {
      "veiculo": {"modelo", "ano_modelo", "data_nf", "data_venda", "data_entrega",
                  "data_emplacamento", "data_garantia"},
      "visitas": [{"data", "abertura", "fechamento", "revisao", "km", "concessionaria",
                   "fonte", "agendado", "codigo_servico"}, ...],
      "volume_loja_12m": 812.0
    }
a última visita da lista é a que acabou de fechar. o volume da loja vem do core,
que enxerga o histórico da concessionária inteira.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features_revisao as fr  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
MODELO = RAIZ / "models" / "modelo_evasao.joblib"
COLS_DATA = ["data", "abertura", "fechamento", "data_nf", "data_venda", "data_entrega",
             "data_emplacamento", "data_garantia"]


@lru_cache(maxsize=1)
def carregar(caminho: str | None = None):
    caminho = Path(caminho) if caminho else MODELO
    meta = json.loads(caminho.with_name("metadata.json").read_text(encoding="utf-8"))
    return joblib.load(caminho), meta


def faixa_prioridade(score, faixas: dict) -> np.ndarray:
    score = np.asarray(score)
    return np.select([score >= faixas["CRITICA"], score >= faixas["ALTA"], score >= faixas["MEDIA"]],
                     ["CRITICA", "ALTA", "MEDIA"], "BAIXA")


def perfil_sugerido(features: pd.DataFrame, score, limiar: float) -> np.ndarray:
    """liga o score aos 4 perfis do PrevioPLS. regra de negócio, não é saída do modelo."""
    score = np.asarray(score)
    atrasou = (features["atraso_plano_dias"] > 60) | (features["km_excedente"] > 3000)
    return np.select(
        [score < limiar, features["revisoes_puladas"] > 0, atrasou],
        ["FIEL", "ECONOMICO", "ESQUECIDO"],
        "ABANDONO",
    )


def historico_para_visitas(req: dict) -> pd.DataFrame:
    visitas = pd.DataFrame(req["visitas"])
    for campo, valor in req["veiculo"].items():
        visitas[campo] = valor
    visitas["vin"] = "consulta"
    visitas["volume_loja_12m"] = float(req["volume_loja_12m"])
    for c in COLS_DATA:
        visitas[c] = pd.to_datetime(visitas[c], errors="coerce")
    visitas["km"] = pd.to_numeric(visitas["km"], errors="coerce")
    visitas["revisao"] = pd.to_numeric(visitas["revisao"])
    visitas["agendado"] = visitas["agendado"].astype(bool)
    return visitas.sort_values("data", kind="stable").reset_index(drop=True)


def classificar(req: dict, caminho_modelo: str | None = None) -> dict:
    modelo, meta = carregar(caminho_modelo)
    visitas = historico_para_visitas(req)
    # modelo fora da lista do treino cai no mesmo balde dos raros
    visitas["modelo"] = visitas["modelo"].where(visitas["modelo"].isin(meta["modelos_conhecidos"]), "OUTROS")

    feats = fr.montar_features(visitas)
    atual = feats.iloc[[-1]]
    score = float(modelo.predict_proba(atual[meta["features"]])[0, 1])
    return {
        "probabilidade_evasao": round(score, 4),
        "prioridade": str(faixa_prioridade([score], meta["faixas"])[0]),
        "gera_lead": score >= meta["limiar"],
        "perfil_sugerido": str(perfil_sugerido(atual, [score], meta["limiar"])[0]),
        "proxima_revisao_estimada": (visitas["data"].iloc[-1]
                                     + pd.Timedelta(days=float(atual["dias_ate_proxima"].fillna(365).iloc[0]))
                                     ).strftime("%Y-%m-%d"),
        "versao_modelo": meta["versao"],
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("uso: python src/inferencia.py requisicao.json")
    requisicao = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(json.dumps(classificar(requisicao), ensure_ascii=False, indent=2))
