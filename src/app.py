from pathlib import Path
import pandas as pd
import streamlit as st
from joblib import load

MODEL_PATH = Path("models") / "pl_model.joblib"
DATA_PATH = Path("data") / "processed" / "matches_features.csv"

st.set_page_config(page_title="Premier League Match Predictor", layout="centered")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    bundle = load(MODEL_PATH)
    return bundle["model"], bundle["features"]

def latest_team_form(df, team_name, side_prefix):
    team_matches = df[(df["home_team"] == team_name) | (df["away_team"] == team_name)].copy()
    team_matches = team_matches.dropna(subset=[
        "home_pts_last5","away_pts_last5",
        "home_gf_last5","away_gf_last5",
        "home_ga_last5","away_ga_last5",
        "home_gd_last5","away_gd_last5"
    ])
    if team_matches.empty:
        return None

    last_row = team_matches.iloc[-1]

    if side_prefix == "home":
        return {
            "home_pts_last5": last_row["home_pts_last5"] if last_row["home_team"] == team_name else last_row["away_pts_last5"],
            "home_gf_last5": last_row["home_gf_last5"] if last_row["home_team"] == team_name else last_row["away_gf_last5"],
            "home_ga_last5": last_row["home_ga_last5"] if last_row["home_team"] == team_name else last_row["away_ga_last5"],
            "home_gd_last5": last_row["home_gd_last5"] if last_row["home_team"] == team_name else last_row["away_gd_last5"],
        }
    else:
        return {
            "away_pts_last5": last_row["home_pts_last5"] if last_row["home_team"] == team_name else last_row["away_pts_last5"],
            "away_gf_last5": last_row["home_gf_last5"] if last_row["home_team"] == team_name else last_row["away_gf_last5"],
            "away_ga_last5": last_row["home_ga_last5"] if last_row["home_team"] == team_name else last_row["away_ga_last5"],
            "away_gd_last5": last_row["home_gd_last5"] if last_row["home_team"] == team_name else last_row["away_gd_last5"],
        }

st.title("Premier League Result Predictor")
st.write("Predict Home win, Draw, or Away win using recent team form from the last 5 matches.")

df = load_data()
model, FEATURES = load_model()

teams = sorted(pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True)))

home_team = st.selectbox("Home team", teams)
away_team = st.selectbox("Away team", teams)

predict_btn = st.button("Predict")

if home_team == away_team:
    st.warning("Please select two different teams.")
elif not predict_btn:
    st.info("Select two teams and click Predict to see probabilities and features.")

if predict_btn and home_team != away_team:
    home_form = latest_team_form(df, home_team, "home")
    away_form = latest_team_form(df, away_team, "away")

    if home_form is None or away_form is None:
        st.error("Not enough history for one of the teams to compute last 5 match form.")
    else:
        row = {**home_form, **away_form}

        row["home_advantage"] = 1
        row["pts_diff_last5"] = row["home_pts_last5"] - row["away_pts_last5"]
        row["gd_diff_last5"] = row["home_gd_last5"] - row["away_gd_last5"]

        X = pd.DataFrame([row])[FEATURES]


        probs = model.predict_proba(X)[0]
        classes = model.classes_
        prob_map = {c: float(p) for c, p in zip(classes, probs)}

        h = prob_map.get("H", 0.0)
        d = prob_map.get("D", 0.0)
        a = prob_map.get("A", 0.0)

        st.subheader("Match")
        st.markdown(f"### {home_team} vs {away_team}")
        st.caption("Inputs are based on rolling form from each team’s last five matches.")

        pred_label, pred_prob = max(
            [("Home win", h), ("Draw", d), ("Away win", a)],
            key=lambda x: x[1]
        )

        if pred_label == "Home win":
            st.success(f"Prediction: {home_team} win  ({pred_prob:.1%})")
        elif pred_label == "Away win":
            st.success(f"Prediction: {away_team} win  ({pred_prob:.1%})")
        else:
            st.info(f"Prediction: Draw  ({pred_prob:.1%})")

        st.subheader("Predicted probabilities")
        c1, c2, c3 = st.columns(3)
        c1.metric("Home win", f"{h:.1%}")
        c2.metric("Draw", f"{d:.1%}")
        c3.metric("Away win", f"{a:.1%}")

        left, right = st.columns([2, 1])

        chart_df = pd.DataFrame(
            {"Probability": [h, d, a]},
            index=["Home win", "Draw", "Away win"]
        )

        with left:
            st.bar_chart(chart_df)

        with right:
            st.write("Probability breakdown")
            st.table(pd.DataFrame({
                "Outcome": ["Home win", "Draw", "Away win"],
                "Probability": [f"{h:.1%}", f"{d:.1%}", f"{a:.1%}"]
            }))

        confidence = max(h, d, a)
        st.write("Model confidence")
        st.progress(confidence)
        st.caption(f"Confidence shows how strongly the model favors its top outcome: {confidence:.1%}")

        st.divider()

        st.subheader("Recent form comparison (last 5 matches)")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Home points", f"{float(X['home_pts_last5'].iloc[0]):.2f}")
        f2.metric("Away points", f"{float(X['away_pts_last5'].iloc[0]):.2f}")
        f3.metric("Home goal diff", f"{float(X['home_gd_last5'].iloc[0]):.2f}")
        f4.metric("Away goal diff", f"{float(X['away_gd_last5'].iloc[0]):.2f}")

        st.subheader("Key signals")
        pts_diff = float(X["pts_diff_last5"].iloc[0]) if "pts_diff_last5" in X.columns else None
        gd_diff = float(X["gd_diff_last5"].iloc[0]) if "gd_diff_last5" in X.columns else None

        if pts_diff is not None:
            if pts_diff > 0:
                st.write(f"Home has stronger recent points form by {pts_diff:.2f} points per match.")
            elif pts_diff < 0:
                st.write(f"Away has stronger recent points form by {abs(pts_diff):.2f} points per match.")
            else:
                st.write("Points form is even over the last five matches.")

        if gd_diff is not None:
            if gd_diff > 0:
                st.write(f"Home has a better recent goal difference by {gd_diff:.2f}.")
            elif gd_diff < 0:
                st.write(f"Away has a better recent goal difference by {abs(gd_diff):.2f}.")
            else:
                st.write("Goal difference is even over the last five matches.")

        st.divider()

        st.subheader("Final result explanation")

        home_pts = float(X["home_pts_last5"].iloc[0])
        away_pts = float(X["away_pts_last5"].iloc[0])
        home_gd = float(X["home_gd_last5"].iloc[0])
        away_gd = float(X["away_gd_last5"].iloc[0])

        if pred_label == "Home win":
            explanation = (
                f"Heading into this matchup, {home_team} look slightly stronger at home. "
                f"The model gives them the top probability at {pred_prob:.1%}, based on recent momentum. "
                f"Over the last five matches, {home_team} average {home_pts:.2f} points per game with a goal difference of {home_gd:.2f}, "
                f"while {away_team} average {away_pts:.2f} points per game with a goal difference of {away_gd:.2f}. "
                f"If both teams play to their recent level, the home side is a bit more likely to take the points."
            )
        elif pred_label == "Away win":
            explanation = (
                f"This looks like a tough away trip for {home_team}. "
                f"The model leans toward {away_team} with a top probability of {pred_prob:.1%}, suggesting their form has been stronger lately. "
                f"Across the last five matches, {away_team} average {away_pts:.2f} points per game with a goal difference of {away_gd:.2f}, "
                f"compared with {home_team} at {home_pts:.2f} points per game and a goal difference of {home_gd:.2f}. "
                f"If the current trend continues, the away side has a slightly better chance of coming away with a win."
            )
        else:
            explanation = (
                f"This matchup looks closely balanced, and the model slightly favors a draw at {pred_prob:.1%}. "
                f"{home_team} average {home_pts:.2f} points per game over their last five matches, while {away_team} average {away_pts:.2f}. "
                f"With both sides showing similar recent output, it is easy to see this one being decided by small moments rather than a clear gap in form."
            )

        st.write(explanation)

        st.caption(
            "The prediction is based on each team’s last five matches and is intended as a guide to recent momentum. "
            "It does not account for injuries, lineups, transfers, or match day context, so outcomes can still differ."
        )

        st.divider()

        with st.expander("Show features used"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("Home form last 5")
                st.write("Points", round(float(X["home_pts_last5"].iloc[0]), 2))
                st.write("Goals for", round(float(X["home_gf_last5"].iloc[0]), 2))
                st.write("Goals against", round(float(X["home_ga_last5"].iloc[0]), 2))
                st.write("Goal diff", round(float(X["home_gd_last5"].iloc[0]), 2))

            with col2:
                st.write("Away form last 5")
                st.write("Points", round(float(X["away_pts_last5"].iloc[0]), 2))
                st.write("Goals for", round(float(X["away_gf_last5"].iloc[0]), 2))
                st.write("Goals against", round(float(X["away_ga_last5"].iloc[0]), 2))
                st.write("Goal diff", round(float(X["away_gd_last5"].iloc[0]), 2))

            st.write("Raw feature row")
            st.table(X)

        with st.expander("What influences home wins in this model"):
            coef_path = Path("models") / "home_win_coefficients.csv"
            if coef_path.exists():
                coef_df = pd.read_csv(coef_path).head(10)
                st.bar_chart(coef_df.set_index("feature"))
                st.caption("Positive values increase the chance of a home win. Negative values reduce it.")
            else:
                st.info("Coefficient file not found. Run training to generate models/home_win_coefficients.csv.")
