import argparse
import os
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from review_insights.baseline import train
from review_insights.data_gen import make_dataset, make_hard
from review_insights.llm_classifier import classify

DATA = Path(__file__).parent / "data"

def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3),
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-in", type=float, default=4.0)
    parser.add_argument("--price-out", type=float, default=20.0)
    parser.add_argument("--llm-sample", type=int, default=60, help="nº de avaliações do teste enviadas ao LLM")
    args = parser.parse_args()

    df = pd.read_csv(DATA / "reviews.csv") if (DATA / "reviews.csv").exists() else make_dataset()
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=0, stratify=df["sentiment"])
    hard = pd.read_csv(DATA / "hard.csv") if (DATA / "hard.csv").exists() else make_hard()
    model = train(train_df["review"], train_df["sentiment"])

    rows = []
    for name, subset in [("teste (sintético)", test_df.head(args.llm_sample)), ("difícil (manual)", hard)]:
        t0 = time.perf_counter()
        pred = model.predict(subset["review"])
        rows.append({"modelo": "TF-IDF+LogReg", "conjunto": name, **metrics(subset["sentiment"], pred),
                     "latência_s": round(time.perf_counter() - t0, 4), "custo_usd": 0.0})

    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        for name, subset in [("teste (sintético)", test_df.head(args.llm_sample)), ("difícil (manual)", hard)]:
            t0 = time.perf_counter()
            result = classify(subset["review"].tolist())
            cost = (result.input_tokens * args.price_in + result.output_tokens * args.price_out) / 1e6
            rows.append({"modelo": "LLM", "conjunto": name, **metrics(subset["sentiment"], result.sentiments),
                         "latência_s": round(time.perf_counter() - t0, 2), "custo_usd": round(cost, 4)})
    else:
        print("(sem credenciais da API: rodando só o baseline)\n")

    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
