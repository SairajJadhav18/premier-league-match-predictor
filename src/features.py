from pathlib import Path
import pandas as pd

IN_PATH = Path("data") / "processed" / "matches_clean.csv"
OUT_PATH = Path("data") / "processed" / "matches_features.csv"

WINDOW = 5

def build_long_format(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def outcome(hg, ag):
        if hg > ag:
            return "H"
        if hg < ag:
            return "A"
        return "D"

    df["y"] = [outcome(hg, ag) for hg, ag in zip(df["home_goals"], df["away_goals"])]

    def pts(side: str, y: str) -> int:
        if y == "D":
            return 1
        if side == "home" and y == "H":
            return 3
        if side == "away" and y == "A":
            return 3
        return 0

    home = pd.DataFrame({
        "match_id": df.index,
        "team": df["home_team"],
        "side": "home",
        "gf": df["home_goals"],
        "ga": df["away_goals"],
        "pts": [pts("home", y) for y in df["y"]]
    })

    away = pd.DataFrame({
        "match_id": df.index,
        "team": df["away_team"],
        "side": "away",
        "gf": df["away_goals"],
        "ga": df["home_goals"],
        "pts": [pts("away", y) for y in df["y"]]
    })

    long_df = pd.concat([home, away], ignore_index=True)
    long_df["gd"] = long_df["gf"] - long_df["ga"]
    return df, long_df

def add_rolling_team_form(df: pd.DataFrame, long_df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    long_df = long_df.sort_values(["team", "match_id"]).reset_index(drop=True)

    for col in ["pts", "gf", "ga", "gd"]:
        long_df[f"{col}_last{window}"] = (
            long_df.groupby("team")[col]
            .apply(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    home_feat = long_df[long_df["side"] == "home"][[
        "match_id",
        f"pts_last{window}",
        f"gf_last{window}",
        f"ga_last{window}",
        f"gd_last{window}"
    ]].rename(columns={
        f"pts_last{window}": "home_pts_last5",
        f"gf_last{window}": "home_gf_last5",
        f"ga_last{window}": "home_ga_last5",
        f"gd_last{window}": "home_gd_last5"
    })

    away_feat = long_df[long_df["side"] == "away"][[
        "match_id",
        f"pts_last{window}",
        f"gf_last{window}",
        f"ga_last{window}",
        f"gd_last{window}"
    ]].rename(columns={
        f"pts_last{window}": "away_pts_last5",
        f"gf_last{window}": "away_gf_last5",
        f"ga_last{window}": "away_ga_last5",
        f"gd_last{window}": "away_gd_last5"
    })

    out = df.reset_index().rename(columns={"index": "match_id"})
    out = out.merge(home_feat, on="match_id", how="left")
    out = out.merge(away_feat, on="match_id", how="left")

    out["home_advantage"] = 1
    out["pts_diff_last5"] = out["home_pts_last5"] - out["away_pts_last5"]
    out["gd_diff_last5"] = out["home_gd_last5"] - out["away_gd_last5"]

    return out

def main():
    df = pd.read_csv(IN_PATH)

    base_df, long_df = build_long_format(df)
    feat = add_rolling_team_form(base_df, long_df, window=WINDOW)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Saving to:", OUT_PATH.resolve())
    print("Columns being saved:")
    print(feat.columns.tolist())

    feat.to_csv(OUT_PATH, index=False)

    print("Saved features to", OUT_PATH)

if __name__ == "__main__":
    main()
