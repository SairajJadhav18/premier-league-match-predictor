Author

Sairaj Jadhav, Sairaj.Jadhav@dal.ca
Computer Science Student at Dalhousie University

Premier League Match Outcome Predictor

This project is an end to end machine learning application that predicts the outcome of Premier League football matches as a Home win, Draw, or Away win.
The prediction is based on historical match data and each team’s recent form over their last five matches.

The goal of this project is to demonstrate how machine learning can be applied to real world sports data, from data processing and feature engineering to model training, evaluation, and deployment through a simple web interface.

What this project does

Given two teams playing a Premier League match, the system:

• Analyzes how both teams have performed in their last five matches
• Computes rolling form statistics such as points, goals scored, goals conceded, and goal difference
• Uses a trained machine learning model to estimate the probability of each possible outcome
• Displays the results in an interactive web app with charts, confidence indicators, and a human readable explanation

The output is not a guaranteed prediction, but a data driven estimate based on recent trends.

Technologies used

This project was built using the following tools and libraries:

• Python
• NumPy
• Pandas
• scikit learn
• Streamlit
• Git and GitHub

Dataset

The dataset used in this project contains historical Premier League match results, including:

• Home team and away team
• Goals scored by each team
• Match result
• Season information

The data was sourced from Kaggle and processed locally for feature engineering and model training.

Feature engineering

To capture recent team performance, the project uses rolling form features calculated over the previous five matches for each team.

Examples of features include:

• Average points per match
• Average goals scored
• Average goals conceded
• Average goal difference
• Home advantage indicator
• Difference in form between home and away teams

These features help the model understand momentum and relative team strength rather than relying only on raw historical outcomes.

Machine learning models

Two machine learning models were trained and compared:

• Logistic Regression
• Random Forest Classifier

To ensure realistic evaluation, a season based train test split was used.
Older seasons are used for training, while future seasons are reserved for testing.
This prevents data leakage and better simulates real world prediction scenarios.

Model performance was evaluated using:

• Accuracy
• Macro F1 score
• Confusion matrices

The best performing model was saved and used by the Streamlit application.

Model interpretability

To improve transparency, the project analyzes feature coefficients from the Logistic Regression model.
This helps explain which factors increase or decrease the likelihood of a home win.

These insights are visualized in the application to help users understand why a prediction was made.

Streamlit web application

An interactive Streamlit app allows users to:

• Select a home team and away team
• View predicted probabilities for all outcomes
• See confidence levels for the prediction
• Compare recent form between teams
• Read a natural language explanation of the predicted result

This makes the model accessible even to users without a machine learning background.

Project structure
premier_league_predictor/
│
├── src/
│   ├── download_data.py      # Data download and loading
│   ├── data_prep.py          # Data cleaning and preprocessing
│   ├── features.py           # Feature engineering
│   ├── train.py              # Model training and evaluation
│   └── app.py                # Streamlit application
│
├── models/
│   └── pl_model.joblib       # Trained machine learning model
│
├── data/
│   ├── raw/                  # Raw datasets (ignored by Git)
│   └── processed/            # Processed datasets (ignored by Git)
│
├── README.md
└── .gitignore


Large data files and trained models are excluded from version control for cleanliness.

How to run the project locally
1. Install dependencies

Make sure Python is installed, then run:

pip install pandas numpy scikit-learn streamlit joblib

2. Generate features and train the model
python src/features.py
python src/train.py


This will generate features, train the models, and save the best performing model.

3. Run the Streamlit app
streamlit run src/app.py


The app will open in your browser, where you can interactively make predictions.

Limitations

This project does not account for:

• Player injuries
• Starting lineups
• Transfers
• Tactical changes
• Match day conditions

Predictions are based purely on historical data and recent form, so results should be interpreted as guidance rather than certainty.

Learning outcomes

Through this project, I gained hands on experience with:

• Data preprocessing and feature engineering
• Preventing data leakage in ML workflows
• Model evaluation and comparison
• Model interpretability
• Deploying machine learning models using Streamlit
• Structuring a clean, professional GitHub project

