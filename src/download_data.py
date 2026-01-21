import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "zaeemnalla/premier-league",
    ""
)

print("First 5 rows")
print(df.head())

print("\nColumns")
print(df.columns)

print("\nDataset info")
print(df.info())

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/matches.csv", index=False)

print("\nSaved dataset to data/raw/matches.csv")
