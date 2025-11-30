# N3 — Projeto Completo de Ciência de Dados (Mercado Imobiliário)

## Alunos
- **Alberto Zilio**
- **Roni Pereira**

> Disciplina: Ciência de Dados  
> Avaliação: **N3 — Trabalho Final (Projeto completo)**

---

## 1) Problema de Negócio (Parte 1)

Este projeto tem como objetivo analisar dados do **mercado imobiliário** e construir um modelo de **Machine Learning** para apoiar decisões de negócio.

**Pergunta de negócio (versão atual):**  
Estimar o **preço por metro quadrado (`preco_m2`)** de um imóvel a partir de suas características (ex.: área, nº quartos, vagas, taxa condominial, etc.).

**Tipo de problema:** Regressão (variável contínua).

> Observação: caso identifiquemos que a pergunta não está adequada aos dados, a formulação pode ser ajustada.

---

## 2) Estrutura do Repositório

```text
N3_CIENCIA_DADOS/
├── data/
│   ├── dataset_original/
│   │   └── tb_mercadoimob.csv
│   ├── dataset_processado_aulas_anteriores/
│   │   └── dados_atualizados.xlsx
│   └── dataset_processado_N3/
│       ├── base_modelagem.csv
│       └── etl_report.json
├── notebooks/
│   ├── notebooks_aulas_anteriores/
│   │   ├── Aula_01_02_Ciencia_de_Dados_.ipynb
│   │   ├── Aula_03_Ciencia_de_Dados_.ipynb
│   │   ├── Aula_04_Ciencia_de_Dados.ipynb
│   │   ├── Aula_05.ipynb
│   │   └── Aula06_Complemento_Final.ipynb
│   └── notebooks_N3/
│       └── 00_etl_N3_completo.ipynb
├── scripts/
├── modelo_final.pkl
├── requirements.txt
└── README.md

---


## 3) Pipeline e Arquitetura (Parte 2)
### Visão geral do pipeline
**Entrada (RAW):**
- `data/dataset_original/tb_mercadoimob.csv`

**Processamento (ETL):**
- Notebook: `notebooks/notebooks_N3/00_etl_N3_completo.ipynb`

**Saídas (PROCESSED):**
- `data/dataset_processado_N3/base_modelagem.csv`
- `data/dataset_processado_N3/etl_report.json`

### O que foi feito no Passo 01 (ETL)
No notebook `00_etl_N3_completo.ipynb` foram realizados:

1. **Leitura do dataset original**
   - Carregamento do CSV de imóveis.

2. **Tratamento e padronização**
   - Conversão de colunas numéricas (ex.: área, valor, taxas e contagens).
   - Tratamento de valores inválidos (ex.: conversão com `errors='coerce'`).

3. **Engenharia de atributos**
   - Criação do **alvo**:  
     - `preco_m2 = valor / area` (com validações para evitar divisão por zero e valores inválidos).
   - Criação de variáveis binárias (quando aplicável):
     - `vista_mar_bin` (busca por termos no título/descrição)
     - `mobiliado_bin` (busca por termos no título/descrição)

4. **Filtros e qualidade**
   - Manutenção de registros com `area > 0` e `valor > 0`.
   - Filtragem para casos de **Venda** (quando o dataset possui a coluna de tipo de negócio).
   - Remoção de duplicidades (por `hash` ou `id`, se disponível).
   - Tratamento de outliers no `preco_m2` com estratégia estável (ex.: clipping por IQR).

5. **Geração da base final para modelagem**
   - Criação e salvamento de uma base “limpa” e pronta para treino:
     - `data/dataset_processado_N3/base_modelagem.csv`
   - Geração do relatório do ETL:
     - `data/dataset_processado_N3/etl_report.json` (shape, features usadas, target, observações)

> Resultado: o Passo 01 foi executado com sucesso e os arquivos foram gerados corretamente.

---

## 4) Modelagem e Avaliação Comparativa (Parte 3) — (Próximo passo)
**A implementar nos notebooks N3:**
- `01_eda_N3.ipynb` (EDA final resumida para o relatório)
- `02_modelagem_N3.ipynb` (treino e comparação)

Requisitos:
- Treinar **pelo menos 3 modelos** (ex.: Regressão Linear/Ridge, Random Forest, Gradient Boosting).
- Avaliar com **pelo menos 3 métricas** (ex.: MAE, RMSE, R²).
- Apresentar **tabela comparativa** e justificar o melhor modelo.

---

## 5) Deploy (Parte 4) — (Próximo passo)
Requisitos:
- Salvar o modelo escolhido em `modelo_final.pkl` (pickle/joblib).
- Carregar o modelo salvo e prever um **novo exemplo** (nunca visto), descrevendo a previsão.

A organizar em `/scripts`:
- `scripts/train.py` (treino + salvar modelo)
- `scripts/predict.py` (carregar + prever uma entrada exemplo)

---

## 6) Como executar o projeto
### 6.1 Criar ambiente virtual (recomendado)
**Windows (PowerShell):**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
