from pathlib import Path
import pandas as pd

RAW_PATH = Path("data") / "raw" / "results.csv"
OUT_PATH = Path("data") / "processed" / "matches_clean.csv"

def main():
    df = pd.read_csv(RAW_PATH)

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    df = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals", "result"])

    df = df.sort_values("season").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("Saved cleaned matches to", OUT_PATH)

if __name__ == "__main__":
    main()
