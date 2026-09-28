Nomes e RMs,
Giovanne Charelli Zaniboni Silva | 556223
Gustavo Oliveira de Moura | 555827
Lynn Bueno Rosa | 551102

# PrevioPLS · IA & ML · Sprint 3

Previsão de evasão pós-venda na rede Ford (Desafio 02 · VIN Share), feita sobre a planilha oficial do desafio `vin_share_Desafio_02.xlsx`.

**Pergunta:** quando um veículo termina uma revisão na rede oficial, ele volta para a próxima revisão no prazo?
**Resposta do modelo:** uma probabilidade de evasão por visita, que o PrevioPLS transforma em lead priorizado para o consultor.

## Resultado em uma olhada

| | |
|---|---|
| Base | 602.788 linhas → 534.424 visitas de 175 mil veículos, serviços de 01/2020 a 05/2026 |
| Alvo | evasão = não voltar à rede até a próxima revisão (10 mil km ou 12 meses) + 90 dias · 34,4% das visitas |
| Modelos comparados | baseline, regressão logística, KNN, árvore de decisão, random forest, XGBoost e MLP |
| Modelo escolhido | **XGBoost** (melhor na validação cruzada e no teste, menor sobreajuste, mais estável no tempo) |
| Teste temporal (visitas a partir de 07/2024, 122 mil) | **ROC-AUC 0,698 · PR-AUC 0,547** (acaso: 0,356) |
| No limiar escolhido no treino | encontra **63% das evasões**, com precisão de 51% |
| Na operação | abordar 30% da carteira cobre **46% das evasões**; a faixa crítica evade 64%, a baixa 23% |

## Estrutura

```
PrevioPLS-IAML-Sprint3/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── previopls_evasao_sprint3.ipynb   entrega principal, executado, com todas as saídas
│   └── previopls_evasao_sprint3.html    o mesmo notebook para abrir no navegador
├── docs/
│   ├── documentacao.md                  documentação resumida da solução
│   └── figuras/                         gráficos gerados pelo notebook
├── src/
│   ├── features_revisao.py              features por visita (usada no treino e na inferência)
│   ├── inferencia.py                    classificação de um veículo a partir do histórico
│   └── exemplo_requisicao.json          requisição de exemplo gerada pelo notebook
├── models/
│   ├── modelo_evasao.joblib             pipeline final (pré-processamento + XGBoost)
│   └── metadata.json                    features, limiar, faixas de prioridade e métricas
└── data/                                a planilha da Ford vai aqui (não versionada)
```

## Como rodar

1. Coloque a planilha oficial em `data/vin_share_Desafio_02.xlsx`. Ela não vem no pacote: são 81 MB de dado da Ford, distribuído pelo canal do challenge.
2. Instale as dependências (Python 3.10 ou mais novo):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Abra e execute o notebook:

```bash
jupyter notebook notebooks/previopls_evasao_sprint3.ipynb
```

A leitura da planilha pelo `openpyxl` leva uns 3 minutos e a execução completa, uns 35 minutos numa máquina de 12 núcleos. Para reexecutar mais rápido, salve uma vez a planilha lida (`bruto.to_pickle("vin_share.pkl")`) e aponte a variável de ambiente `VIN_SHARE_CACHE` para esse arquivo. A variável `VIN_SHARE_XLSX` troca o caminho da planilha.

Para só ver os resultados, abra o `.html`: ele tem todas as tabelas e gráficos da última execução.

### Classificar um veículo sem abrir o notebook

```bash
python src/inferencia.py src/exemplo_requisicao.json
```

Devolve probabilidade de evasão, faixa de prioridade (CRÍTICA, ALTA, MÉDIA, BAIXA), se vira lead, perfil sugerido (Fiel, Econômico, Esquecido, Abandono) e a data estimada da próxima revisão.

Antes de carregar, o `inferencia.py` confere o SHA-256 do `modelo_evasao.joblib`: o joblib executa código ao desserializar, e um arquivo trocado rodaria o código de quem trocou. Se retreinar o modelo, atualize `MODELO_SHA256` no mesmo commit (o comando está no comentário da constante).

## O que o notebook cobre

| Critério da sprint | Seção |
|---|---|
| Compreensão do problema | 1 · contexto, objetivo, dicionário de dados, granularidade, caracterização como classificação binária |
| Preparação dos dados | 2 · diagnóstico, limpeza com log, outliers, construção do alvo com censura, features, teste anti-vazamento, divisão temporal, seleção por informação mútua, transformações |
| Desenvolvimento dos modelos | 3 · 7 algoritmos (baseline, regressão logística, KNN, árvore, random forest, XGBoost, MLP) com busca de hiperparâmetros, ajuste fino, ablação e curva de aprendizado |
| Avaliação e comparação | 4 · ROC-AUC, PR-AUC, F1, Brier no teste temporal, limiar, ganho operacional, calibração, segmentos, importância, erros |
| Conclusão | 5 · modelo escolhido, integração com o PrevioPLS, deploy, fila de leads de hoje, melhorias |
