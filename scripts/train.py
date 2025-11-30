from __future__ import annotations

import json
import datetime
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# ----------------------------
# URLs RAW (GitHub)
# ----------------------------
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


def evaluate(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)     # compatível com versões sem `squared=`
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def main() -> None:
    root = find_project_root()
    out_dir = root / "data" / "dataset_processado_N3"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Lendo base_modelagem.csv (GitHub RAW)...")
    df = pd.read_csv(CSV_URL)

    report = {}
    try:
        print("Lendo etl_report.json (GitHub RAW)...")
        report = read_json_from_url(REPORT_URL)
    except Exception as e:
        print("Aviso: não foi possível ler etl_report.json. Motivo:", repr(e))
        report = {}

    target = report.get("target", "preco_m2")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' não encontrada no CSV. Colunas: {df.columns.tolist()}")

    features = report.get("features")
    if not features:
        features = [c for c in df.columns if c != target]

    X = df[features].copy()
    y = df[target].copy()

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge(alpha=1.0)": Ridge(alpha=1.0, random_state=42),
        "RandomForest(n=300)": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
    }

    rows = []
    pipelines = {}

    print("\nTreinando e avaliando modelos...")
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocess),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        m = evaluate(y_test, y_pred)
        rows.append({"Modelo": name, **m})
        pipelines[name] = pipe

        print(f"- {name}: MAE={m['MAE']:.4f} | RMSE={m['RMSE']:.4f} | R2={m['R2']:.4f}")

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

    # salvar tabela comparativa
    out_results = out_dir / "comparacao_modelos.csv"
    results.to_csv(out_results, index=False)
    print("\nTabela comparativa salva em:", out_results)

    # escolher melhor
    best_name = results.loc[0, "Modelo"]
    best_pipe = pipelines[best_name]
    print("Melhor modelo (menor RMSE):", best_name)

    # salvar modelo_final.pkl (com backup)
    model_path = root / "modelo_final.pkl"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_path.exists():
        backup = root / f"modelo_final_backup_{ts}.pkl"
        model_path.replace(backup)
        print("Backup criado:", backup)

    joblib.dump(best_pipe, model_path)
    print("Modelo final salvo em:", model_path)

    # salvar também métricas em json (opcional, mas útil)
    metrics = {
        "best_model": best_name,
        "target": target,
        "metrics": {
            "MAE": float(results.loc[0, "MAE"]),
            "RMSE": float(results.loc[0, "RMSE"]),
            "R2": float(results.loc[0, "R2"]),
        },
        "generated_at": ts,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Métricas salvas em:", metrics_path)


if __name__ == "__main__":
    main()
