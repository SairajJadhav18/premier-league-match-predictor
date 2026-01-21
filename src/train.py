from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

DATA_PATH = Path("data") / "processed" / "matches_features.csv"
MODEL_PATH = Path("models") / "pl_model.joblib"

FEATURES = [
    "home_pts_last5",
    "away_pts_last5",
    "home_gf_last5",
    "away_gf_last5",
    "home_ga_last5",
    "away_ga_last5",
    "home_gd_last5",
    "away_gd_last5",
    "home_advantage",
    "pts_diff_last5",
    "gd_diff_last5",
]

def season_sort_key(s: str):
    # converts "2006-2007" into 2006 for sorting
    try:
        return int(str(s).split("-")[0])
    except Exception:
        return 0

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    if isinstance(model, LogisticRegression):
        class_index = list(model.classes_).index("H")

        coef_df = pd.DataFrame({
            "feature": FEATURES,
            "coefficient": model.coef_[class_index]
        }).sort_values("coefficient", ascending=False)

        print("\nTop features pushing HOME wins")
        print(coef_df.head(10))

        print("\nTop features pushing AGAINST home wins")
        print(coef_df.tail(10))

        # save coefficients for resume and README use
        coef_df.to_csv("models/home_win_coefficients.csv", index=False)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print("\n" + "=" * 60)
    print("Model:", name)
    print("Accuracy:", acc)
    print("Macro F1:", f1)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))
    print("Report:")
    print(classification_report(y_test, preds))

    return acc, f1, model

def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + ["y"]).reset_index(drop=True)

    seasons = sorted(df["season"].unique(), key=season_sort_key)
    if len(seasons) < 3:
        raise ValueError("Not enough seasons to do a season based split.")

    test_seasons = seasons[-2:]
    train_seasons = seasons[:-2]

    train_df = df[df["season"].isin(train_seasons)].copy()
    test_df = df[df["season"].isin(test_seasons)].copy()

    print("Train seasons:", train_seasons[0], "to", train_seasons[-1])
    print("Test seasons:", test_seasons)

    X_train = train_df[FEATURES]
    y_train = train_df["y"]
    X_test = test_df[FEATURES]
    y_test = test_df["y"]

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=4000, class_weight="balanced")),
        ("Random Forest", RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=10,
            random_state=42
        )),
    ]

    results = []
    best = None

    for name, m in models:
        acc, f1, fitted = evaluate_model(name, m, X_train, y_train, X_test, y_test)
        results.append((name, acc, f1, fitted))

    best = sorted(results, key=lambda x: (x[2], x[1]), reverse=True)[0]
    best_name, best_acc, best_f1, best_model = best

    print("\n" + "=" * 60)
    print("Best model:", best_name)
    print("Best Accuracy:", best_acc)
    print("Best Macro F1:", best_f1)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": best_model, "features": FEATURES, "model_name": best_name}, MODEL_PATH)
    print("Saved best model to", MODEL_PATH)

if __name__ == "__main__":
    main()
