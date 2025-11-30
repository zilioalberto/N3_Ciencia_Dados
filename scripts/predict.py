from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import joblib


# URLs RAW (GitHub)
CSV_URL = "https://raw.githubusercontent.com/zilioalberto/N3_Ciencia_Dados/main/data/dataset_processado_N3/base_modelagem.csv"
REPORT_URL = "https://raw.githubusercontent.com/zilioalberto/N3_Ciencia_Dados/main/data/dataset_processado_N3/etl_report.json"


def read_json_from_url(url: str) -> dict:
    with urlopen(url) as resp:
        content = resp.read().decode("utf-8")
    return json.loads(content)


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "requirements.txt").exists() or (p / "README.md").exists() or (p / "data").exists():
            return p
    return start


def main() -> None:
    root = find_project_root()
    model_path = root / "modelo_final.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Não encontrei modelo_final.pkl na raiz do projeto. Rode primeiro: python scripts/train.py")

    # carregar modelo
    model = joblib.load(model_path)

    # carregar report para target/features (se falhar, usa fallback)
    report = {}
    try:
        report = read_json_from_url(REPORT_URL)
    except Exception:
        report = {}

    target = report.get("target", "preco_m2")
    features = report.get("features")

    # carregar base para obter um exemplo real (evita erro de schema)
    df = pd.read_csv(CSV_URL)
    if not features:
        features = [c for c in df.columns if c != target]

    # exemplo: pegar 1 linha aleatória e prever
    example = df[features].iloc[[0]].copy()
    pred = float(model.predict(example)[0])

    print("Modelo carregado:", model_path)
    print("Target:", target)
    print("\nExemplo de entrada (1 linha):")
    print(example)
    print(f"\nPrevisão de {target}: {pred:.2f}")


if __name__ == "__main__":
    main()
